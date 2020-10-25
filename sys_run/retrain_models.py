# while mutating
# for pool in cost_tiers
#     for model in pool
#     new_model = mutate(model)   
#       cost = cost(new_model)
#     new_pool = get_cost_tier(cost)
#     new_pool.add(cost, new_model)
# for pool in cost_tiers:
#   pool = keep_topk(pool)
#
import math 
import logging
import sys
import asyncio
import time
import os
import argparse
import glob
import copy
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import sys
sys.path.append(sys.path[0].split("/")[0])
import cost
from cost import ModelEvaluator, CostTiers
import util
from database import Database 
from monitor import Monitor, TimeoutError
from train import train_model
import torch.distributed as dist
import torch.multiprocessing as mp
import ctypes
import datetime
from job_queue import ProcessQueue


class TrainTaskInfo:
  def __init__(self, task_id, model_dir, models, model_id):
    self.task_id = task_id
    self.model_dir = model_dir
    self.models = models
    self.model_id = model_id

 
def init_process(task_id, fn, train_args, gpu_ids, model_id, models, model_dir,\
                train_psnrs, valid_psnrs, log_format, task_logger, backend='nccl'):
  fn(task_id, train_args, gpu_ids, model_id, models, model_dir, \
    train_psnrs, valid_psnrs, log_format, task_logger)



class Trainer():
  def __init__(self, args):
    # build loggers
    self.log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
      format=self.log_format, datefmt='%m/%d %I:%M:%S %p')

    self.train_logger = util.create_logger('train_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'train_log'))
    self.task_logger = util.create_logger('task_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'task_log'))
    self.train_logger.info("args = %s", args)

    self.args = args  
  
    mp.set_start_method("spawn")

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    self.model_manager = util.ModelManager(args.model_path, args.starting_model_id)


  def load_model(self, model_id):
    self.load_monitor.set_error_msg(f"---\nLoading model id {model_id} timed out\n---")
    with self.load_monitor:
      try:
        model_ast = self.model_manager.load_model_ast(model_id)
      except RuntimeError:
        self.load_monitor.logger.info(f"---\nError loading model {model_id}\n---")
        return None
      except TimeoutError:
        return None
      else:
        if has_loop(model_ast):
          self.debug_logger.info(f"Tree has loop!!\n{model_ast.dump()}")
          return None
        return model_ast

              
  def lower_model(self, new_model_id, new_model_ast):
    self.lowering_monitor.set_error_msg(f"---\nfailed to lower model {new_model_id}\n---")
    with self.lowering_monitor:
      try:
        new_models = [new_model_ast.ast_to_model() for i in range(self.args.model_initializations)]
      except TimeoutError:
        return None
      else:
        return new_models


  def create_train_process(self, train_task_info, gpu_ids, train_psnrs, validation_psnrs):
    model_id = train_task_info.model_id
    task_id = train_task_info.task_id
    model_dir = train_task_info.model_dir 
    models = train_task_info.models

    p = mp.Process(target=init_process, args=(task_id, run_train, self.args, gpu_ids, model_id, models, model_dir, \
                                              train_psnrs, validation_psnrs, self.log_format, self.task_logger))
    return p

  def run_training_tasks(self, gpu_ids, process_queue, train_psnrs, validation_psnrs):
    timeout = self.args.train_timeout
    bootup_time = 30
    available_gpus = set((0,1,2,3))
    running_processes = {}
    start_times = {}
    restarted = set()
    failed = set()

    while True:
      if process_queue.is_empty() and len(running_processes) == 0:
        break
      # check for finished tasks and kill any that have exceeded timemout
      running_tasks = [tid for tid in running_processes.keys()]
      self.task_logger.info(f"running tasks {running_tasks}")
      for task_id in running_tasks:
        task, task_info = running_processes[task_id]

        # check if process is done
        if not task.is_alive():
          self.task_logger.info(f"task {task_id} model_id {task_info.model_id} process name {task.name} is finished on time")
          self.task_logger.info(f"tasks still in queue {[tid for tid, t in process_queue.queue]}")
          task.join()
          # mark the GPU it used as free
          available_gpus.add(gpu_ids[task_id])
          del running_processes[task_id]

        else: # task is still alive, check if it timed out
          curr_time = time.time()
          start_time = start_times[task_id]
          if curr_time - start_time > timeout:
            self.task_logger.info(f"task {task_id} model_id {task_info.model_id} process name {task.name} timed out, killing at {datetime.datetime.now()}")
            task.terminate()
            task.join()
            # mark the GPU it used as free
            available_gpus.add(gpu_ids[task_id])
            del running_processes[task_id]
      
          # check if task ran into issues starting up and needs to be restarted
          if curr_time - start_time > bootup_time:
            if not os.path.exists(f"{task_info.model_dir}/v0_train_log"):
              task.terminate()
              task.join()
              if not task_id in restarted:
                self.task_logger.info(f"task {task_id} model_id {task_info.model_id} process name {task.name} " +
                                      f"unresponsive, restarting on gpu {gpu_ids[task_id]}...")
                new_task = self.create_train_process(task_info, gpu_ids, train_psnrs, validation_psnrs)
                new_task.start()
                start_times[task_id] = time.time()
                restarted.add(task_id)
              else: # we've already failed to restart this task, give up 
                failed.add(task_id)
                available_gpus.add(gpu_ids[task_id])
                del running_processes[task_id]
      
      self.task_logger.info(f"available_gpus {available_gpus}")

      # fill up any available gpus
      available_gpu_list = [g for g in available_gpus]
      for available_gpu in available_gpu_list:
        if process_queue.is_empty():
          break
        task, task_info = process_queue.take()
        task_id = task_info.task_id
        gpu_ids[task_id] = available_gpu
        self.task_logger.info(f"starting task {task_id} model_id {task_info.model_id} on gpu {available_gpu}")
        task.start()
        running_processes[task_id] = (task, task_info)
        start_times[task_id] = time.time()
        available_gpus.remove(available_gpu)

      time.sleep(20)

    return failed 


  def save_model_ast(self, model_ast, model_id, model_dir):
    self.save_monitor.set_error_msg(f"---\nfailed to save model {model_id}\n---")
    with self.save_monitor:
      try:
        self.model_manager.save_model_ast(model_ast, model_dir)
        self.model_manager.save_model_info_file(model_dir, self.args.model_initializations)
      except TimeoutError:
        return False
      else:
        return True

  def save_model(self, mutation_batch_info, new_model_id):
    self.save_monitor.set_error_msg(f"---\nfailed to save model {new_model_id}\n---")
    with self.save_monitor:
      try:
        pytorch_models = mutation_batch_info.pytorch_models[new_model_id]
        model_ast = mutation_batch_info.model_asts[new_model_id]
        model_dir = mutation_batch_info.model_dirs[new_model_id]
        self.model_manager.save_model(pytorch_models, model_ast, model_dir)
      except TimeoutError:
        return False
      else:
        return True

  def sample_models_to_retrain(self):
    model_list = []
    with open(self.args.model_retrain_list, "r") as f:
      for l in f:
        model_id = int(l.strip())
        model_list.append(model_id)
    return model_list
    
  # searches over program mutations within tiers of computational cost
  def run(self):
    process_queue = ProcessQueue()

    # importance sample which models from each tier to mutate based on PSNR
    model_ids = self.sample_models_to_retrain()
    mutation_batch_info = MutationBatchInfo() # we'll store results from this mutation batch here          
    training_tasks = []

    size = len(model_ids) * self.args.model_initializations
    valid_psnrs = mp.Array(ctypes.c_double, [-1]*size)
    train_psnrs = mp.Array(ctypes.c_double, [-1]*size)
    gpu_ids = mp.Array(ctypes.c_int, [-1]*len(model_ids))

    for task_id, model_id in enumerate(model_ids):
      model_ast = self.load_model(model_id)
     
      pytorch_models = self.lower_model(model_id, model_ast)
    
      compute_cost = self.evaluator.compute_cost(model_ast)

      model_dir = self.model_manager.model_dir(model_id)
      
      task_info = TrainTaskInfo(task_id, model_dir, pytorch_models, model_id)

      training_tasks.append((model_ast, task_info))
    
    for ast, task_info in training_tasks:
      task = self.create_train_process(task_info, gpu_ids, train_psnrs, valid_psnrs)
      process_queue.add((task, task_info))

    failed_tasks = self.run_training_tasks(gpu_ids, process_queue, training_tasks, valid_psnrs)

    

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--model_path', type=str, help='path with model information and where to save training results')
  parser.add_argument('--save', type=str, help='experiment name')

  parser.add_argument('--train_timeout', type=int, default=2400)

  # training parameters
  parser.add_argument('--num_gpus', type=int, default=4, help='number of available GPUs') # change this to use all available GPUs
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
  parser.add_argument('--lr_search_steps', type=int, default=1, help='how many line search iters for finding learning rate')
  parser.add_argument('--variance_min', type=float, default=0.003, help='minimum validation psnr variance')
  parser.add_argument('--variance_max', type=float, default=0.02, help='maximum validation psnr variance')
  parser.add_argument('--weight_decay', type=float, default=1e-16, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=200, help='training report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='trained weights save frequency')
  parser.add_argument('--epochs', type=int, default=3, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--train_portion', type=int, default=1e5, help='portion of training data to use')
  parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')
  parser.add_argument('--validation_freq', type=int, default=50, help='validation frequency for assessing validation PSNR variance')

  # training full chroma + green parameters
  parser.add_argument('--use_green_input', action="store_true")
  parser.add_argument('--full_model', action="store_true")

  args = parser.parse_args()

  if not torch.cuda.is_available():
    sys.exit(1)

  random.seed(args.seed)
  np.random.seed(args.seed)
  #torch.cuda.set_device(args.gpu)
  cudnn.benchmark = False
  torch.manual_seed(args.seed)

  cudnn.enabled=True
  cudnn.deterministic=True
  torch.cuda.manual_seed(args.seed)

  util.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  args.model_path = os.path.join(args.save, args.model_path)

  trainer = Trainer(args)
  trainer.train(args.cost_tiers, args.tier_size)


