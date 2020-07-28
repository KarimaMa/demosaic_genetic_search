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

sys.path.append(sys.path[0].split("/")[0])

import util
from dataset import GreenDataset
from kcore_util import *

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--model_path', type=str, help='directory where subset models are stored')
  parser.add_argument('--model_version', type=int, help='model version to use')
  parser.add_argument('--training_file', type=str, help='a file with list of training data images')
  parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
  parser.add_argument('--samples_per_iter', type=int, default=40000)
  parser.add_argument('--budget', type=int, help='size of kcore set')
  args = parser.parse_args()
  util.create_dir(args.save, scripts_to_save=glob.glob('*.py'))
  args.results_file = os.path.join(args.save, args.results_file)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  experiment_logger = util.create_logger('experiment_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'experiment_log'))
  experiment_logger.info("args = %s", args)

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = False
  torch.manual_seed(args.seed)

  cudnn.enabled=True
  cudnn.deterministic=True
  torch.cuda.manual_seed(args.seed)

  if not torch.cuda.is_available():
    sys.exit(1)

  model_manager = util.ModelManager(args.model_path)
  model, ast = model_manager.load_model('seed', args.model_version)
  kcore_set = kcore_greedy(model, args, experiment_logger)

