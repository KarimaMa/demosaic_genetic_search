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
import demosaicnet_models
from dataset import GreenDataset, Dataset, FastDataLoader


def run(args, models, model_dir):
  print(f"training {len(models)} models")
  log_format = '%(asctime)s %(levelname)s %(message)s'

  if not args.infer:
    train_loggers = [util.create_logger(f'v{i}_train_logger', logging.INFO, \
                                        log_format, os.path.join(model_dir, f'v{i}_train_log'))\
                    for i in range(len(models))]

  validation_loggers = [util.create_logger(f'v{i}_validation_logger', logging.INFO, \
                                          log_format, os.path.join(model_dir, f'v{i}_validation_log')) \
                      for i in range(len(models))]

  test_loggers = [util.create_logger(f'v{i}_test_logger', logging.INFO, \
                                          log_format, os.path.join(model_dir, f'v{i}_test_log')) \
                      for i in range(len(models))]

  model_pytorch_files = [util.get_model_pytorch_file(model_dir, model_version) \
                        for model_version in range(len(models))]

  models = [model.to(device=f"cuda:{args.gpu}") for model in models]
  for m in models:
    m.to_gpu(args.gpu)

  criterion = nn.MSELoss()

  optimizers = [torch.optim.Adam(
      m.parameters(),
      args.learning_rate) for m in models]

  if not args.full_model:
    if not args.infer:
      train_data = GreenDataset(data_file=args.training_file, flatten=False) 
    validation_data = GreenDataset(data_file=args.validation_file, flatten=False)
    test_data = GreenDataset(data_file=args.test_file, flatten=False)
  else:
    if not args.infer:
      train_data = Dataset(data_file=args.training_file, flatten=False)
    validation_data = Dataset(data_file=args.validation_file, flatten=False)
    test_data = Dataset(data_file=args.test_file, flatten=False)

  if not args.infer:
    num_train = len(train_data)
    train_indices = list(range(int(num_train*args.train_portion)))
 
    train_queue = FastDataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
        pin_memory=True, num_workers=8)

  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))
  num_test = len(test_data)
  test_indices = list(range(num_test))

  validation_queue = FastDataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
      pin_memory=True, num_workers=8)

  test_queue = FastDataLoader(
    test_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SequentialSampler(test_indices),
    pin_memory=True, num_workers=8)

  if not args.infer:
    for epoch in range(args.epochs):
      train_losses = train_epoch(args, train_queue, models, criterion, optimizers, train_loggers, \
        model_pytorch_files, validation_queue, validation_loggers, epoch)
      print(f"finished epoch {epoch}")
      valid_losses = infer(args, validation_queue, models, criterion, validation_loggers)
      test_losses = infer(args, test_queue, models, criterion, test_loggers)

      for i in range(len(models)):
        print(f"valid loss {valid_losses[i]}")
        validation_loggers[i].info('validation epoch %03d %e', epoch, valid_losses[i])
        test_loggers[i].info('test epoch %03d %e', epoch, test_losses[i])
  else:
    valid_losses, val_psnr = infer(args, validation_queue, models, criterion, validation_loggers)
    test_losses, test_psnr = infer(args, test_queue, models, criterion, test_loggers)

    for i in range(len(models)):
      print(f"valid loss {valid_losses[i]}")
      validation_loggers[i].info('validation %e my psnr %e michael psnr %e', \
          valid_losses[i], util.compute_psnr(valid_losses[i]), val_psnr)
      test_loggers[i].info('test %e my psnr %e michael psnr %e', \
          test_losses[i], util.compute_psnr(test_losses[i]), test_psnr)

    return valid_losses, test_losses 

  return valid_losses, train_losses


def train_epoch(args, train_queue, models, criterion, optimizers, train_loggers, model_pytorch_files, validation_queue, validation_loggers, epoch):
  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.train()

  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)
    input = input.cuda()
    target = target.cuda()

    for i, model in enumerate(models):
      optimizers[i].zero_grad()
      pred = model(input)
      loss = criterion(pred, target)

      if args.testing:
        print("pred")
        print(pred[0,:,0:4,0:4])
        print("target")
        print(target[0,:,0:4,0:4])
        exit()

      loss.backward()
      optimizers[i].step()

      loss_trackers[i].update(loss.item(), n)

      if step % args.report_freq == 0 or step == len(train_queue)-1:
        train_loggers[i].info('train %03d %e', epoch*len(train_queue)+step, loss.item())

    if step % args.save_freq == 0 or step == len(train_queue)-1:
      for i in range(len(models)):
        torch.save(models[i].state_dict(), model_pytorch_files[i])

    if not args.validation_freq is None and step % args.validation_freq == 0:
      valid_losses = infer(args, validation_queue, models, criterion, validation_loggers)
      for i in range(len(models)):
        print(f"validation loss {valid_losses[i]}")
        validation_loggers[i].info(f'validation {epoch*len(train_queue)+step} {valid_losses[i]}')

  return [loss_tracker.avg for loss_tracker in loss_trackers]


def infer(args, valid_queue, models, criterion, validation_loggers):
  running_psnr = torch.tensor(0)
  running_count = torch.tensor(0)

  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda()
   
    n = input.size(0)

    for i, model in enumerate(models):
      pred = model(input)
      clamped = torch.clamp(pred, min=0, max=1)

      loss = criterion(clamped, target)
      loss_trackers[i].update(loss.item(), n)

      count = torch.tensor(pred.shape[0])  # account for batch_size > 1
      mse = (clamped-target).square().mean(-1).mean(-1).mean(-1)
      psnr = -10.0*torch.log10(mse)
      # average per-image psnr
      running_count = running_count + count
      running_psnr = running_psnr + (psnr.sum(0) - count.float()*running_psnr) / running_count.float() 

  return [loss_tracker.avg for loss_tracker in loss_trackers], running_psnr


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--momentum', type=float, default=90., help='momentum')
  parser.add_argument('--weight_decay', type=float, default=1e-32, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='save frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')
  parser.add_argument('--training_file', type=str, default='/home/karima/cnn-data/subset7_100k_train_files.txt', help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, default='/home/karima/cnn-data/subset7_100k_val_files.txt', help='filename of file with list of validation data image files')
  parser.add_argument('--test_file', type=str, default='/home/karima/cnn-data/test_files.txt', help='filename of file with list of test data image files')

  parser.add_argument('--results_file', type=str, default='training_results', help='where to store training results')
  parser.add_argument('--validation_freq', type=int, default=None, help='validation frequency')
  parser.add_argument('--layers', type=int)
  parser.add_argument('--width', type=int)
  parser.add_argument('--full_model', action='store_true')
  parser.add_argument('--testing', action='store_true')

  parser.add_argument('--infer', action='store_true')
  parser.add_argument('--pretrained', action='store_true')
  parser.add_argument('--weights', type=str, default="/home/karima/demosaicnet/demosaicnet/data/bayer.pth")

  args = parser.parse_args()

  args.model_path = os.path.join(args.save, args.model_path)
  args.results_file = os.path.join(args.save, args.results_file)
  model_manager = util.ModelManager(args.model_path, 0)
  model_dir = model_manager.model_dir('seed')
  util.create_dir(model_dir)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  logger = util.create_logger('training_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'training_log'))
  logger.info("args = %s", args)

  if args.infer:
    args.model_initializations = 1

  if args.full_model:
    print("using full demosaicknet")
    models = [demosaicnet_models.FullDemosaicknet(depth=args.layers, width=args.width).cuda() \
              for i in range(args.model_initializations)]
  else:
    models = [demosaicnet_models.GreenDemosaicknet(depth=args.layers, width=args.width).cuda() \
              for i in range(args.model_initializations)]

  for n,p in models[0].named_parameters():
    print(p.shape)

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

  for m in models:
    m._initialize_parameters()
    
  if args.full_model:
    out_c = 3
  else:
    out_c = 1
  
  #print(f"compute cost : {models[0].compute_cost()}")

  model_manager.save_model(models, None, model_dir)

  if args.pretrained:
    state_dict = torch.load(args.weights)
    models[0].load_state_dict(state_dict)
 
  validation_losses, training_losses = run(args, models, model_dir) 

  model_manager.save_model(models, None, model_dir)

  with open(args.results_file, "a+") as f:
    training_losses = [str(tl) for tl in training_losses]
    training_losses_str = ",".join(training_losses)

    validation_losses = [str(vl) for vl in validation_losses]
    validation_losses_str = ",".join(validation_losses)

    data_string = f"training losses: {training_losses_str} validation losses: {validation_losses_str}\n"

    f.write(data_string)


