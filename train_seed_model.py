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

import util
import meta_model
import model_lib
from torch_model import ast_to_model
from dataset import GreenDataset

def train(args, models, model_id, model_dir):
  print("training")
  print(len(models))
  log_format = '%(asctime)s %(levelname)s %(message)s'
  train_loggers = [util.create_logger(f'{model_id}_v{i}_train_logger', logging.INFO, \
                                      log_format, os.path.join(model_dir, f'v{i}_train_log'))\
                  for i in range(len(models))]

  validation_loggers = [util.create_logger(f'{model_id}_validation_logger', logging.INFO, \
                                          log_format, os.path.join(model_dir, f'v{i}_validation_log')) \
                      for i in range(len(models))]

  model_pytorch_files = [util.get_model_pytorch_file(model_dir, model_version) \
                        for model_version in range(len(models))]

  models = [model.cuda() for model in models]

  criterion = nn.MSELoss()

  optimizers = [torch.optim.Adam(
      m.parameters(),
      args.learning_rate,
      weight_decay=args.weight_decay) for m in models]

  train_data = GreenDataset(args.training_file) 
  validation_data = GreenDataset(args.validation_file)

  num_train = len(train_data)
  train_indices = list(range(int(num_train*args.train_portion)))
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


  for epoch in range(args.epochs):
    # training
    train_losses = train_epoch(train_queue, models, criterion, optimizers, train_loggers, model_pytorch_files)
    # validation
    valid_losses = infer(validation_queue, models, criterion)
    for i in range(len(models)):
      validation_loggers[i].info('validation epoch %03d %e', epoch, valid_losses[i])

  return valid_losses


def train_epoch(train_queue, models, criterion, optimizers, train_loggers, model_pytorch_files):
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
      momentum=args.momentum,
      optimizers[i].step()

      loss_trackers[i].update(loss.item(), n)

    if step % args.report_freq == 0:
      for i in range(len(models)):
        train_loggers[i].info('train %03d %e', step, loss_trackers[i].avg)

    if step % args.save_freq == 0:
      for i in range(len(models)):
        torch.save(models[i], model_pytorch_files[i])

  return [loss_tracker.avg for loss_tracker in loss_trackers]


def infer(valid_queue, models, criterion):
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
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--momentum', type=float, default=90., help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='save frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--default_channels', type=int, default=16, help='num of output channels for conv layers')
  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--save', type=str, default='SEARCH', help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')
  parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')

  args = parser.parse_args()

  args.model_path = os.path.join(args.save, args.model_path)
  model_manager = util.ModelManager(args.model_path)
  model_dir = model_manager.model_dir('seed')
  util.create_dir(model_dir)

  full_model = meta_model.MetaModel()
  full_model.build_default_model() 
  green = full_model.green
  #green = model_lib.multires_green_model()
  green.compute_input_output_channels()

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

  valid_losses = train(args, models, 'seed', model_dir) 
  # only save the best model
  min_validation_loss = min(valid_losses)
  best_model_version = valid_losses.index(min_validation_loss)
  model_manager.save_model([ models[best_model_version] ], green, model_dir)

  """
  reloaded_model_dir = model_manager.model_dir('reloaded')
  util.create_dir(reloaded_model_dir)
  reloaded_model, reloaded_ast = model_manager.load_model('seed', 0)
  print(reloaded_ast.dump())
  valid_losses = train(args, [reloaded_model], 'reloaded', reloaded_model_dir)
  """

