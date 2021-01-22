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
import shutil
sys.path.append(sys.path[0].split("/")[0])
from cost import ModelEvaluator
import util
from database import Database 
from monitor import Monitor, TimeoutError
from train import run_model, train_model
from demosaic_ast import load_ast, get_green_model_id, Input
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


def run_train(task_id, train_args, gpu_ids, model_id, pytorch_models, model_dir, \
              train_psnrs, valid_psnrs, log_format, task_logger):
  inits = train_args.model_initializations
  gpu_id = gpu_ids[task_id]
  try:
    for m in pytorch_models:
      m._initialize_parameters()
  except RuntimeError:
    debug_logger.debug(f"Failed to initialize model {model_id}")
    print(f"Failed to initialize model {model_id}")
  else:
    util.create_dir(model_dir)
    training_logger = util.create_logger(f'model_{model_id}_train_logger', logging.INFO, log_format, \
                                        os.path.join(model_dir, f'model_{model_id}_training_log'))
    
    print('Task ', task_id, ' launched on GPU ', gpu_id, ' model id ', model_id)
    if train_args.infer:
      model_valid_psnrs = run_model(train_args, gpu_id, model_id, pytorch_models, model_dir, training_logger)
    else:
      model_valid_psnrs, model_train_psnrs = train_model(train_args, gpu_id, model_id, pytorch_models, model_dir, training_logger)
    
    for i in range(train_args.model_initializations):
      index = train_args.model_initializations * task_id + i 
      if not train_args.infer:
        train_psnrs[index] = model_train_psnrs[i]
      valid_psnrs[index] = model_valid_psnrs[i]


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
    self.model_evaluator = ModelEvaluator(args)

    mp.set_start_method("spawn", force=True)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'


  def load_model(self, model_id):
    try:
      model_ast = self.model_manager.load_model_ast(model_id)
    except RuntimeError:
      return None
    except TimeoutError:
      return None
    else:
      return model_ast


  """
  inserts the green model ast referenced by the green_model_id stored in 
  an input node into the input node.
  NOTE: WE MAKE SURE ONLY ONE GREEN MODEL DAG IS CREATED SO THAT 
  IT IS ONLY RUN ONCE PER RUN OF THE FULL MODEL
  """
  def insert_green_model(self, new_model_ast, green_model=None, green_model_weight_file=None):
    if green_model is None:
      green_model_id = get_green_model_id(new_model_ast)
      green_model_ast_file = self.args.green_model_asts[green_model_id]
      green_model_weight_file = self.args.green_model_weights[green_model_id]

      green_model = demosaic_ast.load_ast(green_model_ast_file)

    nodes = new_model_ast.preorder()
    for n in nodes:
      if type(n) is Input:
        if n.name == "Input(GreenExtractor)":
          n.node = green_model
          n.weight_file = green_model_weight_file
        elif hasattr(n, "node"): # other input ops my run submodels that also require the green model
          self.insert_green_model(n.node, green_model, green_model_weight_file)
  
  def set_no_grad(self, model):
    nodes = model.preorder()
    for n in nodes:
      if type(n) is demosaic_ast.Input:
        if n.name == "Input(GreenExtractor)":
          n.no_grad = no_grad            
        elif hasattr(n, "node"): # other input ops my run submodels that also
          n.no_grad = no_grad
          set_no_grad(n.node, no_grad)


  def lower_model(self, new_model_id, new_model_ast):
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


  def get_models_to_retrain(self):
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
    model_ids = self.get_models_to_retrain()
    training_tasks = []

    size = len(model_ids) * self.args.model_initializations
    valid_psnrs = mp.Array(ctypes.c_double, [-1]*size)
    train_psnrs = mp.Array(ctypes.c_double, [-1]*size)
    gpu_ids = mp.Array(ctypes.c_int, [-1]*len(model_ids))

    for task_id, model_id in enumerate(model_ids):
      model_info_dir = os.path.join(self.args.model_info_dir, f"{model_id}")
      model_ast_file = os.path.join(model_info_dir, "model_ast")
      model_ast = load_ast(model_ast_file)
     
      if self.args.full_model:
        self.insert_green_model(model_ast) # inserts pretrained weights and the chosen green model ast
        self.set_no_grad(model_ast) # sets no_grad parameter for all submodels

      compute_cost = self.model_evaluator.compute_cost(model_ast)

      pytorch_models = self.lower_model(model_id, model_ast)
      
      save_model_dir = os.path.join(self.args.model_path, f"{model_id}")
      util.create_dir(save_model_dir)
      
      shutil.copyfile(model_ast_file, os.path.join(save_model_dir, "model_ast"))

      task_info = TrainTaskInfo(task_id, save_model_dir, pytorch_models, model_id)

      training_tasks.append((model_ast, task_info))
    
    for ast, task_info in training_tasks:
      task = self.create_train_process(task_info, gpu_ids, train_psnrs, valid_psnrs)
      process_queue.add((task, task_info))

    failed_tasks = self.run_training_tasks(gpu_ids, process_queue, training_tasks, valid_psnrs)

    

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--model_path', type=str, default='models', help='where to save training results')
  parser.add_argument('--model_info_dir', type=str, help='where model asts are stored')
  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--model_retrain_list', type=str, help='filename with list of model ids to retrain')
  parser.add_argument('--train_timeout', type=int, default=3600)
  parser.add_argument('--crop', type=int, default=16)

  # training parameters
  parser.add_argument('--seed', type=int, default=2)
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
  parser.add_argument('--validation_variance_start_step', type=int, default=400, help='training step from which to start sampling validation PSNR for assessing variance')
  parser.add_argument('--validation_variance_end_step', type=int, default=1600, help='training step from which to start sampling validation PSNR for assessing variance')

  # training full chroma + green parameters
  parser.add_argument('--full_model', action="store_true")
  parser.add_argument('--rgb8chan', action="store_true")

  parser.add_argument('--green_model_asts', type=str, help="file with list of filenames of green model asts")
  parser.add_argument('--green_model_weights', type=str, help="file with list of filenames of green model weights")

  parser.add_argument('--infer', action='store_true')
  parser.add_argument('--no_grad', action='store_true', help='whether or not to backprop through green model')

  args = parser.parse_args()

  if not torch.cuda.is_available():
    sys.exit(1)

  if args.full_model:
    args.green_model_asts = [l.strip() for l in open(args.green_model_asts)]
    args.green_model_weights = [l.strip() for l in open(args.green_model_weights)]


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
  trainer.run()


