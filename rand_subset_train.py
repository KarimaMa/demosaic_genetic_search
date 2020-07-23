import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import logging
import argparse
import glob
import os
import sys
import random 
import numpy as np

import util
import meta_model
import model_lib
from torch_model import ast_to_model
from dataset import GreenDataset
from train_seed_model import train

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
  parser.add_argument('--save_freq', type=float, default=500, help='save frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--epochs', type=int, default=5, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--save', type=str, default='RAND_DATA_SUBSET_SELECTION', help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--train_portion', type=float, default=0.05, help='portion of training data')
  parser.add_argument('--training_subsets', type=str, help='a file with list of training_files for each subset:')
  parser.add_argument('--validation_file', type=str, help='name of file with list of subsets of validation image files')
  parser.add_argument('--results_file', type=str, default='training_results', help='where to store training results')

  args = parser.parse_args()
  util.create_dir(args.save, scripts_to_save=glob.glob('*.py'))
  args.results_file = os.path.join(args.save, args.results_file)
  args.model_path = os.path.join(args.save, args.model_path)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  experiment_logger = util.create_logger('experiment_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'experiment_log'))
  experiment_logger.info("args = %s", args)

  model_manager = util.ModelManager(args.model_path)

  full_model = meta_model.MetaModel()
  full_model.build_default_model() 
  green = full_model.green
  green.compute_input_output_channels()
  #green = model_lib.multires_green_model()

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = False
  torch.manual_seed(args.seed)

  cudnn.enabled=True
  cudnn.deterministic=True
  torch.cuda.manual_seed(args.seed)

  with open(args.training_subsets, "r") as f:
    training_subsets = [l.strip() for l in f]

  for i, subset in enumerate(training_subsets):
    print(f"training models on subset {i}")
    args.training_file = subset
    model_dir = model_manager.model_dir(f'subset_{i}')
    util.create_dir(model_dir)

    if not torch.cuda.is_available():
      sys.exit(1)

    models = [green.ast_to_model().cuda() for i in range(args.model_initializations)]

    for m in models:
      m._initialize_parameters()

    validation_losses, training_losses = train(args, models, f'subset_{i}', model_dir) 
    model_manager.save_model(models, green, model_dir)

    with open(args.results_file, "a+") as f:
      training_losses = [str(tl) for tl in training_losses]
      training_losses_str = ",".join(training_losses)

      validation_losses = [str(vl) for vl in validation_losses]
      validation_losses_str = ",".join(validation_losses)

      data_string = f"subset {i} training losses: {training_losses_str} validation_losses: {validation_losses_str}\n"

      f.write(data_string)

