import logging
import math
import sys
import asyncio
import time
import os
import argparse
import glob
from types import SimpleNamespace
import copy
import random
import numpy as np
import torch
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from cost import ModelEvaluator
import demosaic_ast
from demosaic_ast import get_green_model_id, set_green_model_id
from mutate import Mutator, has_downsample, MutationType
import util
from database import Database 
from train import train_model
import torch.distributed as dist
import torch.multiprocessing as mp
import ctypes
import datetime
import mysql_db
from queue import Empty



class TrainTaskInfo:
  def __init__(self, task_id, model_dir, model_id, compute_cost, models, training_args):
    self.task_id = task_id
    self.model_dir = model_dir
    self.model_id = model_id
    self.compute_cost = compute_cost
    self.models = models
    self.training_args = training_args


"""
 worker subprocess function that runs training
 gpu_id: the gpu assigned to this worker
"""
def run_train(work_q, pending_q, pending_l, finished_tasks, gpu_id, log_format, logger):
  # keep pulling from work queue
  while True:
    try:
      train_task_info = work_q.get(block=False)
      if train_task_info is None:
        work_q.task_done()
        return

      pending_l.acquire()
      pending_q.put((train_task_info, time.time()))
      pending_l.release()

      model_id = train_task_info.model_id
      task_id = train_task_info.task_id
      model_dir = train_task_info.model_dir 
      models = train_task_info.models
      training_args = train_task_info.training_args

      logger.info(f'---worker on gpu {gpu_id} running task {train_task_info.task_id} model {train_task_info.model_id} learning rate {training_args.learning_rate} ---')
      print(f'--- worker on gpu {gpu_id} running model {train_task_info.model_id} learning rate {training_args.learning_rate} ---')
    
      for m in models:
        m._initialize_parameters()

      print(f"model dir {model_dir}")
     
      train_logger = util.create_logger(f'model_{model_id}_train_logger', logging.INFO, log_format, \
                                  os.path.join(model_dir, f'model_{model_id}_training_log'))

      model_val_psnrs = train_model(training_args, gpu_id, model_id, models, model_dir, train_logger)
      
      logger.info(f"worker on gpu {gpu_id} with task {task_id} validation psnrs: {model_val_psnrs}")

      pending_l.acquire()
      done_task = pending_q.get()
      pending_l.release()
      pending_q.task_done()
      work_q.task_done()

      start_time = done_task[1]
      print(f"task {done_task[0].task_id} took {time.time() - start_time}")
      finished_tasks[done_task[0].task_id] = time.time() - start_time # store time task took

    except Empty:
      time.sleep(10)



class Analyzer():
  def __init__(self, args):
    # build loggers
    self.log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
      format=self.log_format, datefmt='%m/%d %I:%M:%S %p')
    self.logger = util.create_logger('lr_analysis_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'lr_analysis_log'))
    self.work_manager_logger = util.create_logger('work_manager_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'work_manager_log'))
    self.logger.info("args = %s", args)
    self.args = args  
    
    mp.set_start_method("spawn", force=True)

    self.num_workers = self.args.num_gpus
    self.work_queue = mp.JoinableQueue()
    self.pending_queues = [mp.JoinableQueue() for w in range(self.num_workers)]
    self.pending_locks = [mp.Lock() for w in range(self.num_workers)]
    self.work_loggers = [util.create_logger('work_logger', logging.INFO, self.log_format, \
                        os.path.join(args.save, f'work_log_{i}')) for i in range(self.num_workers)]

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    self.evaluator = ModelEvaluator(args)


  def load_model(self, model_id):
    model_ast = demosaic_ast.load_ast(model_id)
    return model_ast


  def insert_green_model(self, model_ast, green_model=None, green_model_weight_file=None):
    if green_model is None:
      green_model_id = get_green_model_id(model_ast)
      green_model_ast_file = self.args.green_model_asts[green_model_id]
      green_model_weight_file = self.args.green_model_weights[green_model_id]

      green_model = demosaic_ast.load_ast(green_model_ast_file)

    nodes = model_ast.preorder()
    for n in nodes:
      if type(n) is demosaic_ast.Input:
        if n.name == "Input(GreenExtractor)":
          n.node = green_model
          n.weight_file = green_model_weight_file
        elif hasattr(n, "node"): # other input ops my run submodels that also require the green model
          self.insert_green_model(n.node, green_model, green_model_weight_file)


  def lower_model(self, model_ast):
    models = [model_ast.ast_to_model() for i in range(self.args.model_initializations)]
    return models


  def create_worker(self, worker_id):
    gpu_id = worker_id
    worker = mp.Process(target=run_train, args=(self.work_queue, self.pending_queues[worker_id], self.pending_locks[worker_id], \
                                                self.finished_tasks, gpu_id, self.log_format, self.work_loggers[worker_id]))
    return worker


  def monitor_training_tasks(self):
    timeout = self.args.train_timeout
    if self.args.ram:
      bootup_time = 300
    else:
      bootup_time = 30
    
    alive = set()
    failed_tasks = []
    for wid, worker in enumerate(self.workers):
      worker.start()
      alive.add(wid)

    tick = 0
    while True:
      if tick % 30 == 0:
        self.work_manager_logger.info(f"alive workers: {alive}")
        print(f"work queue size {self.work_queue.qsize()}")

      if self.work_queue.empty():
        self.work_manager_logger.info("No more models in work queue, waiting for all tasks to complete")

      for wid, worker in enumerate(self.workers): 
        worker_pending_queue = self.pending_queues[wid]

        if not worker.is_alive() and (wid in alive):
          worker.join()
          alive.remove(wid)
          self.work_manager_logger.info(f"worker {wid} is dead with exit code {worker.exitcode}")

          if not worker_pending_queue.empty():
            pending_task = worker_pending_queue.get()
            worker_pending_queue.task_done()
            failed_tasks.append(pending_task[0])

        else: # check if worker has run out of time on current task
          self.pending_locks[wid].acquire()
          if not worker_pending_queue.empty():
            pending_task = worker_pending_queue.get()
            
            start_time = pending_task[1]
            curr_time = time.time()

            terminated = False 

            if curr_time - start_time > timeout:
              self.work_manager_logger.info(f"worker {wid} timed out, killing at {datetime.datetime.now()}")
              worker.terminate()
              worker.join()
              terminated = True

            if terminated:
              failed_tasks.append(pending_task[0])
              worker_pending_queue.task_done()
              new_worker = self.create_worker(wid)
              new_worker.start()
              self.workers[wid] = new_worker
            else:
              worker_pending_queue.put(pending_task)
          
          self.pending_locks[wid].release()
      
      if len(alive) == 0:
        assert self.work_queue.empty(), "all workers are dead with work left in the queue"
        self.work_queue.join()
        self.work_manager_logger.info("All tasks are done")
        break

      time.sleep(10)
      tick += 1

    return failed_tasks


  def save_model_ast(self, model_ast, model_dir):
    ast_file = os.path.join(model_dir, "model_ast")
    model_ast.save_ast(ast_file)

  # searches over program mutations within tiers of computational cost
  def run(self):
    task_id = 0
    self.finished_tasks = mp.Array(ctypes.c_double, [-1]*self.args.sample_n*len(self.args.learning_rates))
    self.training_tasks = []
    self.workers = [self.create_worker(worker_id) for worker_id in range(self.num_workers)]

    # args relevant for training 
    train_argnames = ["crop", "ram", "lazyram", "epochs", "weight_decay", \
                    "training_file", "validation_file", "report_freq", \
                    "save_freq", "validation_freq", "validation_variance_start_step", \
                    "validation_variance_end_step", "batch_size", "seed", \
                    "full_model", "rgb8chan", "model_initializations"]

    training_args = SimpleNamespace()
    for argname in vars(self.args):
      if argname in train_argnames:
        setattr(training_args, argname, getattr(self.args, argname))

    # sample models to train 
    model_ids = [m for m in os.listdir(self.args.modeldir)]
    sampled_model_ids = random.sample(model_ids, self.args.sample_n)

    for model_id in sampled_model_ids:
      self.logger.info(f"loading model {model_id}")
      model_ast = self.load_model(os.path.join(self.args.modeldir, f'{model_id}/model_ast'))
      # get rid of models without parameters
      if not model_ast.has_parameters():
        continue

      if self.args.full_model:
        self.insert_green_model(model_ast)

      pytorch_models = self.lower_model(model_ast)

      # green model must be inserted before computing cost if we're running full model search
      compute_cost = self.evaluator.compute_cost(model_ast)

      model_dir = os.path.join(args.save, f'{model_id}')
      
      for lr in self.args.learning_rates:
        task_args = copy.copy(training_args)
        setattr(task_args, "learning_rate", lr)
        task_info = TrainTaskInfo(task_id, model_dir, model_id, compute_cost, pytorch_models, task_args)
        
        util.create_dir(model_dir)
        self.save_model_ast(model_ast, task_info.model_dir)
    
        self.work_queue.put(task_info)
        self.training_tasks.append(task_info)
        task_id += 1
        
    for w in range(self.num_workers):
      self.work_queue.put(None) # place sentinels telling workers to exit

    failed_tasks = self.monitor_training_tasks()

    print("finished tasks")
    # update model database 
    for task_id, task_time in enumerate(self.finished_tasks):
      if task_time < 0: 
        continue # this task was not run
      task_info = self.training_tasks[task_id]
      task_id = task_info.task_id 
      print(f"task_id {task_id} model id {task_info.model_id} finished")
    
    self.work_queue.join()



if __name__ == "__main__":
  parser = argparse.ArgumentParser("LR Analysis")
  parser.add_argument('--seed', type=int, default=1, help='random seed')

  parser.add_argument('--crop', type=int, default=16, help='how much to crop images during training and inference')

  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--modeldir', type=str, help='where models are stored')
  parser.add_argument('--sample_n', type=int, help='how many models to sample for training')

  parser.add_argument('--deterministic', action='store_true', help="set model psnrs to be deterministic")
  parser.add_argument('--ram', action='store_true')
  parser.add_argument('--lazyram', action='store_true')

  # training full chroma + green parameters
  parser.add_argument('--full_model', action="store_true")
  parser.add_argument('--rgb8chan', action="store_true")

  parser.add_argument('--green_model_asts', type=str, help="file with list of filenames of green model asts")
  parser.add_argument('--green_model_weights', type=str, help="file with list of filenames of green model weights")

  parser.add_argument('--train_timeout', type=int, default=2400)

  # training parameters
  parser.add_argument('--num_gpus', type=int, default=4, help='number of available GPUs') # change this to use all available GPUs
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rates', type=str, default="0.0015,0.003,0.006,0.012", help='list of learning rates to test')

  parser.add_argument('--weight_decay', type=float, default=1e-16, help='weight decay')
  parser.add_argument('--epochs', type=int, default=3, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of initializations to train per model for the first epoch')

  parser.add_argument('--train_size', type=int, default=1e5, help='size of training set')
  parser.add_argument('--training_file', type=str, default="/home/karima/cnn-data/sample_files.txt", help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, default="/home/karima/cnn-data/sample_files.txt", help='filename of file with list of validation data image files')
  
  parser.add_argument('--validation_freq', type=int, default=0.01, help='frequency for assessing validation PSNR variance in terms of percent of training dataset seen')

  args = parser.parse_args()

  args.learning_rates = [float(lr.strip()) for lr in args.learning_rates.split(",")]
  args.report_freq = math.ceil((args.train_size * 0.1) / args.batch_size)
  args.save_freq = math.ceil((args.train_size * 0.5) / args.batch_size)
  args.validation_freq = math.ceil((args.train_size * args.validation_freq) / args.batch_size)
  args.validation_variance_start_step = 0
  args.validation_variance_end_step = args.train_size // args.batch_size

  print(f"validation frequency: {args.validation_freq}")

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  if args.full_model:
    args.green_model_asts = [l.strip() for l in open(args.green_model_asts)]
    args.green_model_weights = [l.strip() for l in open(args.green_model_weights)]
    args.task_out_c = 3
  elif args.rgb8chan:
    args.task_out_c = 3
  else:
    args.task_out_c = 1

  util.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  analyzer = Analyzer(args)
  analyzer.run()



