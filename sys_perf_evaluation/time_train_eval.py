import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import logging
import argparse
import os
import random 
import numpy as np
import time
import sys

sys.path.append(sys.path[0].split("/")[0])

import util
import meta_model
import model_lib
from torch_model import ast_to_model
from dataset import GreenDataset, ids_from_file


def train(args, models):
  log_format = '%(asctime)s %(levelname)s %(message)s'
  train_logger = util.create_logger(f'train_logger', logging.INFO, \
                                      log_format, os.path.join(args.save, f'train_log'))

  models = [model.cuda() for model in models]
  criterion = nn.MSELoss()

  optimizers = [torch.optim.Adam(
      m.parameters(),
      args.learning_rate,
      weight_decay=args.weight_decay) for m in models]

  full_data_filenames = ids_from_file(args.training_file)
  data_filenames = full_data_filenames[0:args.train_portion]
  train_data = GreenDataset(data_filenames=data_filenames, RAM=args.use_ram) 

  num_train = args.train_portion
  train_indices = list(range(args.train_portion))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=True, num_workers=0)

  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.train()

  t0 = time.time()

  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    for i, model in enumerate(models):
      optimizers[i].zero_grad()
      pred = model.run(input)
      loss = criterion(pred, target)

      loss.backward()
      optimizers[i].step()

      loss_trackers[i].update(loss.item(), n)
    
    if step == args.timed_batches:
      t1 = time.time()
      break

  printstr = f"time to train {args.timed_batches} batches: {t1 - t0}"
  print(printstr)
  train_logger.info(printstr)

  return [loss_tracker.avg for loss_tracker in loss_trackers]


def infer(args, models):
  validation_logger = util.create_logger(f'inference_logger', logging.INFO, \
                                          log_format, os.path.join(args.save, f'inference_log')) 

  models = [model.cuda() for model in models]

  criterion = nn.MSELoss()

  data_filenames = ids_from_file(args.validation_file)
  validation_data = GreenDataset(data_filenames=data_filenames, RAM=args.use_ram)
  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))

  validation_queue = torch.utils.data.DataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
      pin_memory=True, num_workers=0)

  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.eval()

  t0 = time.time()
  for step, (input, target) in enumerate(validation_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()
    n = input.size(0)

    for i, model in enumerate(models):
      pred = model.run(input)
      loss = criterion(pred, target)
      loss_trackers[i].update(loss.item(), n)

    if step == args.timed_batches:
      t1 = time.time()
      break

  printstr = f"time to infer {args.timed_batches} batches: {t1 - t0}"
  print(printstr)
  validation_logger.info(printstr)

  return [loss_tracker.avg for loss_tracker in loss_trackers]


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--use_ram', action='store_true')
  parser.add_argument('--train_portion', type=int, help='how many images to use')
  parser.add_argument('--timed_batches', type=int, help="how many batches to time")
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--momentum', type=float, default=90., help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--multires_model', action='store_true')
  parser.add_argument('--demosaicnet', action='store_true')
  parser.add_argument('--ahd', action='store_true')
  parser.add_argument('--ahd2d', action='store_true')
  parser.add_argument('--basic_model2d', action='store_true')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')
  parser.add_argument('--save', type=str, help='path to save experiment data')

  args = parser.parse_args()

  util.create_dir(args.save)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  logger = util.create_logger('experiment_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'experiment_log'))
  logger.info("args = %s", args)

  if args.multires_model:
    logger.info("TRAINING MULTIRES GREEN")
    green = model_lib.multires_green_model()
  elif args.demosaicnet:
    logger.info("TRAINING DEMOSAICNET GREEN")
    green = model_lib.mini_demosaicnet()
  elif args.ahd:
    logger.info(f"TRAINING AHD GREEN")
    green = model_lib.ahd1D_green_model()
  elif args.ahd2d:
    logger.info(f"TRAINING AHD2D GREEN")
    green = model_lib.ahd2D_green_model()
  elif args.basic_model2d:
    logger.info(f"TRAINING BASIC_MODEL2D GREEN")
    green = model_lib.basic2D_green_model()
  else:
    logger.info("TRAINING BASIC GREEN")
    full_model = meta_model.MetaModel()
    full_model.build_default_model() 
    green = full_model.green

  if not torch.cuda.is_available():
    sys.exit(1)

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = False
  torch.manual_seed(args.seed)

  cudnn.enabled=True
  cudnn.deterministic=True
  torch.cuda.manual_seed(args.seed)

  models = [green.ast_to_model().cuda() for i in range(args.model_initializations)]
  for m in models:
    m._initialize_parameters()
    
  training_losses = train(args, models) 
  validation_losses = infer(args, models)

