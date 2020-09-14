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
import sys

sys.path.append(sys.path[0].split("/")[0])

import util
import meta_model
import model_lib
from torch_model import ast_to_model
from dataset import GreenDataset, ids_from_file
from validation_variance_tracker import  VarianceTracker

def get_optimizers(args, models):
  optimizers = [torch.optim.Adam(
      m.parameters(),
      lr=args.learning_rate,
      weight_decay=args.weight_decay) for m in models]
  return optimizers

def create_loggers(model_id, model_dir, num_models, mode):
  log_format = '%(asctime)s %(levelname)s %(message)s'
  log_files = [os.path.join(model_dir, f'v{i}_{mode}_log') for i in range(num_models)]
  for log_file in log_files:
    if os.path.exists(log_file):
      os.remove(log_file)
  loggers = [util.create_logger(f'{model_id}_v{i}_{mode}_logger', logging.INFO, log_format, log_files[i]) for i in range(num_models)]
  return loggers

def train(args, models, model_id, model_dir, experiment_logger):
  print(f"training {len(models)} models")

  train_loggers = create_loggers(model_id, model_dir, len(models), "train")
  validation_loggers = create_loggers(model_id, model_dir, len(models), "validation")

  model_pytorch_files = [util.get_model_pytorch_file(model_dir, model_version) \
                        for model_version in range(len(models))]

  models = [model.cuda() for model in models]

  criterion = nn.MSELoss()
  optimizers = get_optimizers(args, models)

  full_data_filenames = ids_from_file(args.training_file)
  used_filenames = full_data_filenames[0:int(args.train_portion)]
  train_data = GreenDataset(data_filenames=used_filenames, RAM=True) 
  validation_data = GreenDataset(data_file=args.validation_file, RAM=True)

  num_train = len(train_data)
  train_indices = list(range(num_train))
  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=True, num_workers=0)

  validation_queue = torch.utils.data.DataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
      pin_memory=True, num_workers=0)

  losses, validation_variance = get_validation_variance(args, models, model_pytorch_files, criterion, optimizers, train_queue, train_loggers, validation_queue, validation_loggers)
  initial_lr = args.learning_rate
  if validation_variance < args.variance_min:
    args.learning_rate *= 2
  elif validation_variance > args.variance_max:
    args.learning_rate /= 2

  experiment_logger.info(f"INITIAL LEARNING RATE {initial_lr} INDUCES VALIDATION VARIANCE {validation_variance} AT A VALIDATION FREQUENCY OF {args.validation_freq}")
  experiment_logger.info(f"USING LEARNING RATE {args.learning_rate}")

  if initial_lr != args.learning_rate:
    # erase logs from previous epoch and reset models
    train_loggers = create_loggers(model_id, model_dir, len(models), "train")
    validation_loggers = create_loggers(model_id, model_dir, len(models), "validation")
    cur_epoch = 0
    for m in models:
      m._initialize_parameters()
    models = [model.cuda() for model in models]
    optimizers = get_optimizers(args, models)
    
  else: # we can use previous epoch's training 
    cur_epoch = 1

  for epoch in range(cur_epoch, args.epochs):
    train_losses = train_epoch(args, train_queue, models, criterion, optimizers, train_loggers, \
      model_pytorch_files, validation_queue, validation_loggers, epoch)
    valid_losses = infer(args, validation_queue, models, criterion)

    for i in range(len(models)):
      validation_loggers[i].info('validation epoch %03d %e', epoch, valid_losses[i])

  return valid_losses, train_losses


def get_validation_variance(args, models, model_pytorch_files, criterion, optimizers, train_queue, train_loggers, validation_queue, validation_loggers):
  variance_tracker = VarianceTracker(len(models))
  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.train()

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

      if step % args.report_freq == 0 or step == len(train_queue)-1:
        train_loggers[i].info('train %03d %e', step, loss.item())

    if step % args.save_freq == 0 or step == len(train_queue)-1:
      for i in range(len(models)):
        torch.save(models[i].state_dict(), model_pytorch_files[i])

    if step % args.validation_freq == 0 and step > 400:
      valid_losses = infer(args, validation_queue, models, criterion)
      valid_psnrs = [util.compute_psnr(l) for l in valid_losses]
      variance_tracker.update(valid_psnrs)
      
      for i in range(len(models)):
        validation_loggers[i].info(f'validation {step} {valid_losses[i]}')
      
  return [loss_tracker.avg for loss_tracker in loss_trackers], variance_tracker.validation_variance()


def train_epoch(args, train_queue, models, criterion, optimizers, train_loggers, model_pytorch_files, validation_queue, validation_loggers, epoch):
  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.train()

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

      if step % args.report_freq == 0 or step == len(train_queue)-1:
        train_loggers[i].info('train %03d %e', epoch*len(train_queue)+step, loss.item())

    if step % args.save_freq == 0 or step == len(train_queue)-1:
      for i in range(len(models)):
        torch.save(models[i].state_dict(), model_pytorch_files[i])

  return [loss_tracker.avg for loss_tracker in loss_trackers]


def infer(args, valid_queue, models, criterion):
  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()
    n = input.size(0)

    for i, model in enumerate(models):
      pred = model.run(input)
      loss = criterion(pred, target)
      loss_trackers[i].update(loss.item(), n)

  return [loss_tracker.avg for loss_tracker in loss_trackers]


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
  parser.add_argument('--variance_min', type=float, help='minimum validation psnr variance')
  parser.add_argument('--variance_max', type=float, help='maximum validation psnr variance')
  parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='save frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--multires_model', action='store_true')
  parser.add_argument('--multires_model2d', action='store_true')
  parser.add_argument('--demosaicnet', action='store_true')
  parser.add_argument('--ahd', action='store_true')
  parser.add_argument('--ahd2d', action='store_true')
  parser.add_argument('--basic_model2d', action='store_true')
  parser.add_argument('--basic_model', action='store_true')
  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--train_portion', type=int, default=1e5, help='portion of training data to use')
  parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')
  parser.add_argument('--results_file', type=str, default='training_results', help='where to store training results')
  parser.add_argument('--validation_freq', type=int, default=None, help='validation frequency')

  args = parser.parse_args()

  args.model_path = os.path.join(args.save, args.model_path)
  args.results_file = os.path.join(args.save, args.results_file)
  model_manager = util.ModelManager(args.model_path)
  model_dir = model_manager.model_dir('seed')
  util.create_dir(model_dir)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  logger = util.create_logger('training_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'training_log'))
  logger.info("args = %s", args)

  if args.multires_model:
    logger.info("TRAINING MULTIRES GREEN")
    green = model_lib.multires_green_model()
  elif args.multires_model2d:
    logger.info("TRAINING MULTIRES2D GREEN")
    green = model_lib.multires2D_green_model()
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
  elif args.basic_model:
    logger.info("TRAINING BASIC GREEN")
    green = model_lib.basic1D_green_model()

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

  validation_losses, training_losses = train(args, models, 'seed', model_dir, logger) 

  model_manager.save_model(models, green, model_dir)

  with open(args.results_file, "a+") as f:
    training_losses = [str(tl) for tl in training_losses]
    training_losses_str = ",".join(training_losses)

    validation_losses = [str(vl) for vl in validation_losses]
    validation_losses_str = ",".join(validation_losses)

    data_string = f"training losses: {training_losses_str} validation losses: {validation_losses_str}\n"

    f.write(data_string)


