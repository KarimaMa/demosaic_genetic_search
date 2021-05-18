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
from train import train_epoch, infer, get_optimizers, keep_best_model, create_validation_dataset, create_train_dataset, create_loggers

rootdir = '/'.join(sys.path[0].split("/")[0:-1])
sys.path.append(rootdir)

from cost import ModelEvaluator
import util
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
          n.no_grad = False           
        elif hasattr(n, "node"): # other input ops my run submodels that also
          n.no_grad = False
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


  def train_model(self, gpu_id, model_id, models, model_dir, experiment_logger, train_data=None, val_data=None):
    torch.cuda.set_device(gpu_id)
    cudnn.benchmark = False
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    models = [model.to(device=f"cuda:{gpu_id}") for model in models]
    for m in models:
      m.to_gpu(gpu_id)

    train_loggers = create_loggers(model_dir, model_id, len(models), "train")
    validation_loggers = create_loggers(model_dir, model_id, len(models), "validation")
    
    train_queue = create_train_dataset(args, gpu_id, shared_data=train_data, logger=experiment_logger)
    valid_queue = create_validation_dataset(args, gpu_id, shared_data=val_data)

    criterion = torch.nn.MSELoss()

    experiment_logger.info(f"USING LEARNING RATE {self.args.learning_rate}")

    for m in models:
      m._initialize_parameters()
    
    optimizers = get_optimizers(self.args.weight_decay, self.args.learning_rate, models)
    
    for i in range(len(models)):
      train_loggers[i].info(f"STARTING TRAINING AT LR {self.args.learning_rate}") 

    best_val_psnrs = [-1 for i in range(self.args.keep_initializations)]

    for epoch in range(self.args.epochs):
      if args.keep_initializations < self.args.model_initializations:
        if epoch == 1:
          models, train_loggers, validation_loggers, optimizers, model_index = keep_best_model(models, val_psnrs, train_loggers, validation_loggers, optimizers)
          experiment_logger.info(f"after first epoch, validation psnrs {val_psnrs} keeping best model {model_index}")
      
      start_time = time.time()
      train_losses = train_epoch(self.args, gpu_id, train_queue, models, model_dir, criterion, optimizers, train_loggers, \
                                valid_queue, validation_loggers, epoch, start_time)

      model_pytorch_files = [util.get_model_pytorch_file(model_dir, model_version, epoch) for model_version in range(len(models))]
      for i in range(len(models)):
        torch.save(models[i].state_dict(), model_pytorch_files[i])

      end_time = time.time()
      experiment_logger.info(f"time to finish epoch {epoch} : {end_time-start_time}")
      val_losses, val_psnrs = infer(self.args, gpu_id, valid_queue, models, criterion)
      if epoch >= 1:
        best_val_psnrs = [max(best_p, new_p) for (best_p, new_p) in zip(best_val_psnrs, val_psnrs)]

      start_time = time.time()
      for i in range(len(models)):
        validation_loggers[i].info('validation epoch %03d %e %2.3f', epoch, val_losses[i], val_psnrs[i])
      end_time = time.time()
      experiment_logger.info(f"time to validate after epoch {epoch} : {end_time - start_time}")
    return best_val_psnrs


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
      # debug_logger.debug(f"Failed to initialize model {model_id}")
      print(f"Failed to initialize model {model_id}")
    else:
      util.create_dir(save_model_dir)
      training_logger = util.create_logger(f'model_{model_id}_train_logger', logging.INFO, self.log_format, \
                                          os.path.join(save_model_dir, f'model_{model_id}_training_log'))
      
      print('Task ', self.args.task_id, ' launched on GPU ', gpu_id, ' model id ', model_id)
      model_valid_psnrs = self.train_model(gpu_id, model_id, pytorch_models, save_model_dir, training_logger)
      


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
  parser.add_argument('--epochs', type=int, default=6, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--keep_initializations', type=int, default=1, help='number of weight initializations to keep per model after the first epoch')

  parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
 
  parser.add_argument('--report_freq', type=float, default=500, help='training report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='trained weights save frequency')

  parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')

  parser.add_argument('--deterministic', action="store_true")
  parser.add_argument('--green_model_asts', type=str, help="file with list of filenames of green model asts")
  parser.add_argument('--green_model_weights', type=str, help="file with list of filenames of green model weights")

  parser.add_argument('--full_model', action="store_true")
  parser.add_argument('--xtrans_chroma', action="store_true")
  parser.add_argument('--xtrans_green', action="store_true")  
  parser.add_argument('--superres_green', action="store_true")
  parser.add_argument('--superres_rgb', action="store_true")
  parser.add_argument('--superres_only', action="store_true")

  parser.add_argument('--gridsearch', action='store_true')
  parser.add_argument('--nas', action='store_true')
  parser.add_argument('--rgb8chan', action="store_true")

  parser.add_argument('--ram', action='store_true')
  parser.add_argument('--lazyram', action='store_true')

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


