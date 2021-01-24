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
import datetime


def get_model_id(task_id, model_list_file):
  model_ids = [l.strip() for l in open(model_list_file, "r")]
  return model_ids[task_id]


class Trainer():
  def __init__(self, args):
    # build loggers
    self.log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
      format=self.log_format, datefmt='%m/%d %I:%M:%S %p')

    self.train_logger = util.create_logger('train_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'train_log'))
    self.train_logger.info("args = %s", args)

    self.args = args  
    self.model_evaluator = ModelEvaluator(args)


  def insert_green_model(self, new_model_ast, green_model=None, green_model_weight_file=None):
    if green_model is None:
      green_model_id = get_green_model_id(new_model_ast)
      green_model_ast_file = self.args.green_model_asts[green_model_id]
      green_model_weight_file = self.args.green_model_weights[green_model_id]

      green_model = load_ast(green_model_ast_file)

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
      if type(n) is Input:
        if n.name == "Input(GreenExtractor)":
          n.no_grad = self.args.no_grad            
        elif hasattr(n, "node"): # other input ops my run submodels that also
          n.no_grad = self.args.no_grad
          self.set_no_grad(n.node)


  def lower_model(self, new_model_id, new_model_ast):
    try:
      new_models = [new_model_ast.ast_to_model() for i in range(self.args.model_initializations)]
    except TimeoutError:
      return None
    else:
      return new_models


  def get_models_to_retrain(self):
    model_list = []
    with open(self.args.model_retrain_list, "r") as f:
      for l in f:
        model_id = int(l.strip())
        model_list.append(model_id)
    return model_list

    
  def run_train(self):
    model_id = get_model_id(self.args.task_id, self.args.model_retrain_list)

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

    gpu_id = args.gpu_id

    inits = args.model_initializations
    try:
      for m in pytorch_models:
        m._initialize_parameters()

    except RuntimeError:
      debug_logger.debug(f"Failed to initialize model {model_id}")
      print(f"Failed to initialize model {model_id}")
    else:
      util.create_dir(save_model_dir)
      training_logger = util.create_logger(f'model_{model_id}_train_logger', logging.INFO, self.log_format, \
                                          os.path.join(save_model_dir, f'model_{model_id}_training_log'))
      
      print('Task ', self.args.task_id, ' launched on GPU ', gpu_id, ' model id ', model_id)
      if self.args.infer:
        model_valid_psnrs = run_model(self.args, gpu_id, model_id, pytorch_models, save_model_dir, training_logger)
      else:
        model_valid_psnrs, model_train_psnrs = train_model(self.args, gpu_id, model_id, pytorch_models, save_model_dir, training_logger)
      


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--model_path', type=str, default='models', help='where to save training results')
  parser.add_argument('--model_info_dir', type=str, help='where model asts are stored')
  parser.add_argument('--save', type=str, help='where to save training results')
  parser.add_argument('--model_retrain_list', type=str, help='filename with list of model ids to retrain')
  parser.add_argument('--train_timeout', type=int, default=3600)
  parser.add_argument('--crop', type=int, default=16)
  parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use') # change this to use all available GPUs
  parser.add_argument('--task_id', type=int, help='which model in the given list to retrain')

  # training parameters
  parser.add_argument('--seed', type=int, default=2)
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--weight_decay', type=float, default=1e-16, help='weight decay')
  parser.add_argument('--epochs', type=int, default=3, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')

  parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
  parser.add_argument('--lr_search_steps', type=int, default=1, help='how many line search iters for finding learning rate')
  parser.add_argument('--variance_min', type=float, default=0.003, help='minimum validation psnr variance')
  parser.add_argument('--variance_max', type=float, default=0.02, help='maximum validation psnr variance')
  parser.add_argument('--validation_freq', type=int, default=50, help='validation frequency for assessing validation PSNR variance')
  parser.add_argument('--validation_variance_start_step', type=int, default=400, help='training step from which to start sampling validation PSNR for assessing variance')
  parser.add_argument('--validation_variance_end_step', type=int, default=1600, help='training step from which to start sampling validation PSNR for assessing variance')
  
  parser.add_argument('--report_freq', type=float, default=500, help='training report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='trained weights save frequency')

  parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')

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
  trainer.run_train()


