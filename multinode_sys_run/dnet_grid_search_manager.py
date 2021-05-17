import math 
import logging
import time
import os
import argparse
import glob
import copy
import random
import numpy as np
from orderedset import OrderedSet
import torch
import sys

rootdir = '/'.join(sys.path[0].split("/")[0:-1])
sys_run_dir = os.path.join(rootdir, "sys_run")

sys.path.append(rootdir)
sys.path.append(sys_run_dir)

import util
from monitor import Monitor, TimeoutError
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime
import zmq


class Searcher:
  def __init__(self, args):
    self.args = args

  def start_server(self):
    import socket
    hostname = socket.gethostname()

    print("Running manager node on %s" % hostname)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{self.args.port}")

    return socket

  def shutdown_workers(self, socket, num_workers):
    shutdown_time = 60
    start_time = time.time()

    shutdown_workers = []

    while len(shutdown_workers) < num_workers and time.time() - start_time < shutdown_time:
      try:
        message = socket.recv(flags=zmq.NOBLOCK)
        msg_str = message.decode("utf-8")
        
        work_request_info = msg_str.split(" ")
        worker_id = work_request_info[0]
        is_work_request = (int(work_request_info[1]) == 1)

        if not is_work_request:
          print(f"timed out worker {worker_id} sending training results at shutdown!!!")
      
        message_str = "SHUTDOWN"
        print(f"sending shutdown message to {worker_id}")
        reply = message_str.encode("utf-8")
        socket.send(reply)

        shutdown_workers.append(worker_id)

      except zmq.Again as e:
        pass

    if len(shutdown_workers) != num_workers:
      print(f"{num_workers-len(shutdown_workers)} out of {num_workers} were not shutdown")
    else:
      print("all workers successfully shutdown")


  """
  sends the jobs to workers and receive their training results
  """
  def monitor_training_tasks(self, socket, worker_tasks, task_model_ids, validation_psnrs):
    num_tasks = len(worker_tasks)
    
    next_task_id = 0

    pending_task_info = {}
   
    done_tasks = {}

    done_task_ids = OrderedSet()
    timed_out_task_ids = OrderedSet()

    registered_workers = OrderedSet()

    train_argdict = vars(self.args)

    timeout = args.train_timeout + 10
    tick = 0

    while True:
      if tick % 6 == 0:
        print(f"total tasks: {num_tasks}  pending: {len(pending_task_info)}  timed out: {len(timed_out_task_ids)}  done: {len(done_task_ids)}")

      if len(done_task_ids.union(timed_out_task_ids)) == num_tasks:
        print(f"done tasks {done_task_ids} timed out {timed_out_task_ids}")
        self.work_manager_logger.info("-- all tasks are done or timed out --")
        return done_tasks, timed_out_task_ids, registered_workers

      # check for timed out tasks 
      for task_id in pending_task_info:
        start_time = pending_task_info[task_id]["start_time"]
        if time.time() - start_time > timeout:
          if not task_id in timed_out_task_ids:
            timed_out_task_ids.add(task_id)
      
      try:
        # listen for work request from a worker
        message = socket.recv(flags=zmq.NOBLOCK)
        msg_str = message.decode("utf-8")
        
        work_request_info = msg_str.split(" ")
        worker_id = work_request_info[0]

        is_args_request = (int(work_request_info[1]) == 0)
        is_work_request = (int(work_request_info[1]) == 1)

        if not worker_id in registered_workers:
          registered_workers.add(worker_id)

        if is_args_request:
          print(f"Worker {worker_id} has come online, sending training args...")
          socket.send_json(train_argdict)

        elif is_work_request:
          print(f"Received work request from worker {worker_id}")

          # no more work to send - tell worker there is no work
          if next_task_id == num_tasks: 
            message_str = "WAIT"
            print(f"no more work to send to worker {worker_id}, sending {message_str}")
            reply = message_str.encode("utf-8")
            socket.send(reply)

          # send work to worker
          else:
            task_info_str = worker_tasks[next_task_id]
            print(f"sending task {task_info_str} to worker {worker_id}")

            reply = task_info_str.encode("utf-8")
            socket.send(reply)

            pending_task_info[next_task_id] = {"worker": worker_id, "start_time": time.time()}
            next_task_id += 1

        else: # worker is delivering training results
          print(f"Received training results from worker {worker_id}")
          task_id = int(work_request_info[2])
          model_id = int(work_request_info[3])
          task_validation_psnrs = [float(p) for p in work_request_info[4].split(",")]

          if not model_id in task_model_ids:
            print(f"worker sending late results for model {model_id} from previous generation's models, dropping...")
      
          else:          
            print(f"worker {worker_id} on task: {task_id} model_id: {model_id} returning psnrs {task_validation_psnrs}")
            for init in range(args.keep_initializations):
              offset = int(task_id) * args.keep_initializations + init
              validation_psnrs[offset] = task_validation_psnrs[init]
            
            done_tasks[task_id] = time.time() - pending_task_info[task_id]["start_time"]
            done_task_ids.add(task_id)
            del pending_task_info[task_id]

          # send acknowledgement of work received to worker
          ack = "1"
          print(f"sending task ack of work completed to worker {worker_id}")
          reply = ack.encode("utf-8")
          socket.send(reply)
    
      except zmq.Again as e:
        pass

      tick += 1
      time.sleep(5)


  def grid_search(self):
    self.socket = self.start_server()

    if not torch.cuda.is_available():
      sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
   
    layers = [1, 2, 3, 4, 5, 6]
    widths = [8, 16, 32, 64]
    num_models = len(layers)*len(widths)
    task_model_ids = []
    task_id = 0
    subproc_tasks = []
    validation_psnrs = [-1 for i in range(num_models)]


    for l in layers:
      for w in widths:
        model_id = task_id
        model_params = f"{l}-{w}"

        ops = 3*3*l*w*w / 4
        if ops < 5 * 1000:
          lr = 0.004
        elif ops < 15 * 1000:
          lr = 0.002
        elif ops < 30 * 1000:
          lr = 0.001
        else:
          lr = 0.0005

        model_dir = os.path.join(args.model_path, model_params)
        util.create_dir(model_dir)

        subproc_task_str = f"{task_id}--{model_dir}--{model_id}--{lr}"
        task_model_ids.append(model_id)
        
        print(f"adding task {task_id}...")
        subproc_tasks.append(subproc_task_str)
        task_id += 1
    
    done_tasks, timed_out_tasks, registered_workers = self.monitor_training_tasks(self.socket, subproc_tasks, task_model_ids, validation_psnrs)
    num_failed = len(timed_out_tasks)
    num_finished = len(done_tasks)
    minutes_passed = (datetime.datetime.now() - search_start_time).total_seconds() / 60
    print(f"Minutes elapsed: {minutes_passed} Generation: {generation} - successfully trained: {num_finished} - failed train: {num_failed}")
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--port', type=str)

  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--momentum', type=float, default=90., help='momentum')
  parser.add_argument('--weight_decay', type=float, default=1e-32, help='weight decay')

  parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='save frequency')
  parser.add_argument('--epochs', type=int, default=6, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of initializations to train per model for the first epoch')
  parser.add_argument('--keep_initializations', type=int, default=1, help='how many initializations to keep per model after the first epoch')

  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')

  parser.add_argument('--train_portion', type=int, default=1e5, help='portion of training data to use')
  parser.add_argument('--training_file', type=str, default="/home/karima/cnn-data/sample_files.txt", help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, default="/home/karima/cnn-data/sample_files.txt", help='filename of file with list of validation data image files')
 
  parser.add_argument('--crop', type=int, default=0, help='amount to crop output image to remove border effects')
  parser.add_argument('--gridsearch', action='store_true')

  parser.add_argument('--deterministic', action='store_true', help="set model psnrs to be deterministic")
  parser.add_argument('--ram', action='store_true')
  parser.add_argument('--lazyram', action='store_true')

  parser.add_argument('--train_timeout', type=int, default=900)

  parser.add_argument('--num_gpus', type=int, default=4, help='number of available GPUs') # change this to use all available GPUs
  parser.add_argument('--lrsearch', action='store_true', help='whether or not to use lr search')

  args = parser.parse_args()

  args.model_path = os.path.join(args.save, args.model_path)

  os.makedirs(args.model_path, exist_ok=True)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  logger = util.create_logger('training_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'training_log'))
  logger.info("args = %s", args)

  searcher = Searcher(args)
  searcher.grid_search()






