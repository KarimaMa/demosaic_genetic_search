import logging
import os
import sys
import random
from train import train_model

rootdir = '/'.join(sys.path[0].split("/")[0:-1])
sys_run_dir = os.path.join(rootdir, "sys_run")

sys.path.append(rootdir)
sys.path.append(sys_run_dir)

import util
from search_util import insert_green_model
import socket
import zmq
import argparse
import torch
import demosaic_ast
import numpy as np
import time


def parse_task_info_str(task_info_str):
  task_info = task_info_str.split("--")
  task_id = int(task_info[0])
  model_dir = task_info[1]
  model_id = int(task_info[2])

  return task_id, model_id, model_dir


class Args(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

"""
worker process that requests training jobs from manager
"""
def run_worker(args):
  worker_id = args.worker_id
  gpu_id = args.gpu_id

  context = zmq.Context()

  #  Socket to talk to server
  print(f"worker {worker_id} connecting to server {args.host}")
  socket = context.socket(zmq.REQ)
  socket.connect(f"tcp://{args.host}:{args.port}")

  print(f"worker {worker_id} is now online, requesting training args from manager")
  request_str = f"{worker_id} {0}"
  request = request_str.encode("utf-8")
  socket.send(request)

  # wait for manager to send back args
  train_args = Args(socket.recv_json())
  
  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
  logger = util.create_logger('work_logger', logging.INFO, log_format, \
                    os.path.join(train_args.save, f'work_log_{args.worker_id}')) 

  random.seed(train_args.seed)
  np.random.seed(train_args.seed)
  torch.manual_seed(train_args.seed)

  while True:      
    try:
      print("requesting work from manager...")
      request_str = f"{worker_id} {1}"
      request = request_str.encode("utf-8")
      socket.send(request)

      # wait for manager to send work
      message = socket.recv()
      task_info_str = message.decode("utf-8")

      if task_info_str == "WAIT":
        print(f"worker {worker_id} is waiting on work from manager...")
        time.sleep(5)
        continue

      if task_info_str == "SHUTDOWN":
        print("I've seen local minima you people wouldn't believe. Adam firing off the saddle points of loss functions.\n" + \
            "I watched convergence glitter in the dark near the Activation Gates.\n" + \
            "All these update steps will be lost in time, like tears in rain. Time to die.")
        return

      print(f"Received work {task_info_str}")

      task_id, model_id, model_dir = parse_task_info_str(task_info_str)

      model_ast = demosaic_ast.load_ast(os.path.join(model_dir, "model_ast"))

      # get model info
      if train_args.full_model:
        green_model_ast_files = [l.strip() for l in open(train_args.green_model_asts)]
        green_model_weight_files = [l.strip() for l in open(train_args.green_model_weights)]
        insert_green_model(model_ast, green_model_ast_files, green_model_weight_files)

      models = [model_ast.ast_to_model() for i in range(train_args.model_initializations)]

      logger.info(f'---worker {worker_id} running task {task_id} model {model_id} on gpu {gpu_id} ---')
      
      for m in models:
        m._initialize_parameters()
       
      util.create_dir(model_dir)
      train_logger = util.create_logger(f'model_{model_id}_train_logger', logging.INFO, log_format, \
                                  os.path.join(model_dir, f'model_{model_id}_training_log'))

      model_val_psnrs = train_model(train_args, gpu_id, model_id, models, model_dir, train_logger)

      logger.info(f"worker on gpu {gpu_id} finished task {task_id} model {model_id} with validation psnrs {model_val_psnrs}")

      # send training results
      request_str = f"{worker_id} {2} {task_id} {model_id} {','.join([str(p) for p in model_val_psnrs])}"
      request = request_str.encode("utf-8")
      socket.send(request)

      # wait for manager to send acknowledgement of our labors
      message = socket.recv()
      ack = message.decode("utf-8")
      if ack == "SHUTDOWN":
        print(f"Manager sending shutdown signal, ignoring training results for task {task_id} model {model_id}")    
      elif int(ack) == 1:
        print(f"Manager received train results for task {task_id} model {model_id}")
      else:
        print(f"Uh oh... ")
    
    except zmq.Again as e:
      pass

    time.sleep(5)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")

  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--port', type=str)

  parser.add_argument('--host', type=str)
  # parser.add_argument('--seed', type=int, default=1, help='random seed')

  # parser.add_argument('--save', type=str, help='experiment name')

  # seed models 
  # parser.add_argument('--green_seed_model_files', type=str, help='file with list of filenames of green seed model asts')
  # parser.add_argument('--green_seed_model_psnrs', type=str, help='file with list of psnrs of green seed models')

  # parser.add_argument('--nas_seed_model_files', type=str, help='file with list of filenames of nas seed model asts')
  # parser.add_argument('--nas_seed_model_psnrs', type=str, help='file with list of psnrs of nas seed models')

  # parser.add_argument('--rgb8chan_seed_model_files', type=str, help='file with list of filenames of rgb8chan seed model asts')
  # parser.add_argument('--rgb8chan_seed_model_psnrs', type=str, help='file with list of psnrs of rgb8chan seed models')

  # parser.add_argument('--chroma_seed_model_files', type=str, help='file with list of filenames of chroma seed model asts')
  # parser.add_argument('--chroma_seed_model_psnrs', type=str, help='file with list of psnrs of chroma seed models')
  
  # parser.add_argument('--green_model_asts', type=str, help="file with list of filenames of green model asts")
  # parser.add_argument('--green_model_weights', type=str, help="file with list of filenames of green model weights")

  # parser.add_argument('--train_timeout', type=int, default=900)

  # training parameters
  parser.add_argument('--gpu_id', type=int, help="gpu id to use") 
  parser.add_argument('--worker_id', type=int, help='id of this worker node')

  # parser.add_argument('--crop', type=int, help='how much to crop images during training and inference')
  # parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  # parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
  # parser.add_argument('--lrsearch', action='store_true', help='whether or not to use lr search')
  # parser.add_argument('--lr_search_steps', type=int, default=1, help='how many line search iters for finding learning rate')
  # parser.add_argument('--variance_min', type=float, default=0.003, help='minimum validation psnr variance')
  # parser.add_argument('--variance_max', type=float, default=0.02, help='maximum validation psnr variance')
  # parser.add_argument('--weight_decay', type=float, default=1e-16, help='weight decay')
  # parser.add_argument('--report_freq', type=float, default=200, help='training report frequency')
  # parser.add_argument('--save_freq', type=float, default=2000, help='trained weights save frequency')
  # parser.add_argument('--epochs', type=int, default=3, help='num of training epochs')
  # parser.add_argument('--model_initializations', type=int, default=3, help='number of initializations to train per model for the first epoch')
  # parser.add_argument('--keep_initializations', type=int, default=1, help='how many initializations to keep per model after the first epoch')

  # parser.add_argument('--train_portion', type=int, default=1e5, help='portion of training data to use')
  # parser.add_argument('--training_file', type=str, default="/home/karima/cnn-data/sample_files.txt", help='filename of file with list of training data image files')
  # parser.add_argument('--validation_file', type=str, default="/home/karima/cnn-data/sample_files.txt", help='filename of file with list of validation data image files')
  # parser.add_argument('--validation_freq', type=int, default=50, help='validation frequency for assessing validation PSNR variance')
  # parser.add_argument('--validation_variance_start_step', type=int, default=400, help='training step from which to start sampling validation PSNR for assessing variance')
  # parser.add_argument('--validation_variance_end_step', type=int, default=1600, help='training step from which to start sampling validation PSNR for assessing variance')

  # # training full chroma + green parameters
  # parser.add_argument('--full_model', action="store_true")
  # parser.add_argument('--xtrans_green', action="store_true")  
  # parser.add_argument('--superres_green', action="store_true")
  # parser.add_argument('--nas', action='store_true')

  args = parser.parse_args()

  run_worker(args)

