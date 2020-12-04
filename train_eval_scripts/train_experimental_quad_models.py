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
from experimental_green_models import *
from experimental_rgb_models import *
from dataset import GreenQuadDataset, QuadDataset, FastDataLoader


def train(args, models, model_id, model_dir):
  print(f"training {len(models)} models")
  log_format = '%(asctime)s %(levelname)s %(message)s'
  train_loggers = [util.create_logger(f'{model_id}_v{i}_train_logger', logging.INFO, \
                                      log_format, os.path.join(model_dir, f'v{i}_train_log'))\
                  for i in range(len(models))]

  validation_loggers = [util.create_logger(f'{model_id}_v{i}_validation_logger', logging.INFO, \
                                          log_format, os.path.join(model_dir, f'v{i}_validation_log')) \
                      for i in range(len(models))]

  test_loggers = [util.create_logger(f'{model_id}_v{i}_test_logger', logging.INFO, \
                                          log_format, os.path.join(model_dir, f'v{i}_test_log')) \
                      for i in range(len(models))]

  model_pytorch_files = [util.get_model_pytorch_file(model_dir, model_version) \
                        for model_version in range(len(models))]

  models = [model.cuda() for model in models]
    
  criterion = nn.MSELoss()

  optimizers = [torch.optim.Adam(
      m.parameters(),
      args.learning_rate) for m in models]

  if not args.full_model:
    train_data = GreenQuadDataset(data_file=args.training_file) 
    validation_data = GreenQuadDataset(data_file=args.validation_file)
    test_data = GreenQuadDataset(data_file=args.test_file)
  else:
    train_data = QuadDataset(data_file=args.training_file)
    validation_data = QuadDataset(data_file=args.validation_file)
    test_data = QuadDataset(data_file=args.test_file)

  num_train = len(train_data)
  train_indices = list(range(int(num_train*args.train_portion)))
  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))
  num_test = len(test_data)
  test_indices = list(range(num_test))

  train_queue = FastDataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=True, num_workers=8)

  validation_queue = FastDataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SequentialSampler(validation_indices),
      pin_memory=True, num_workers=8)

  test_queue = FastDataLoader(
    test_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SequentialSampler(test_indices),
    pin_memory=True, num_workers=8)
  

  for epoch in range(args.epochs):
    # training
    train_losses = train_epoch(args, train_queue, models, criterion, optimizers, train_loggers, \
      model_pytorch_files, validation_queue, validation_loggers, epoch)
    print(f"finished epoch {epoch}")
    # validation
    valid_losses = infer(args, validation_queue, models, criterion, validation_loggers)
    test_losses = infer(args, test_queue, models, criterion, test_loggers)
    for i in range(len(models)):
      validation_loggers[i].info('validation epoch %03d %e', epoch, valid_losses[i])
      test_loggers[i].info('test epoch %03d %e', epoch, test_losses[i])

  return valid_losses, train_losses


def train_epoch(args, train_queue, models, criterion, optimizers, train_loggers, model_pytorch_files, validation_queue, validation_loggers, epoch):
  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.train()

  for step, (input, target) in enumerate(train_queue):
    bayer_quad = Variable(input, volatile=True).float().cuda()
    target = Variable(target, volatile=True).float().cuda()
    n = bayer_quad.size(0)

    for i, model in enumerate(models):
      optimizers[i].zero_grad()
      
      pred = model(bayer_quad)
      
      if args.testing:
        print(f"mosaic \n{bayer_quad[0,:,0:2,0:2]}")
        print(f"pred \n{pred[0,:,0:4,0:4]}")
        print(f"target \n{target[0,:,0:4,0:4]}")
        exit()

      #loss = criterion(pred[:,1,:,:], target[:,1,:,:]) 
      loss = criterion(pred, target)
      loss.backward()
      optimizers[i].step()

      loss_trackers[i].update(loss.item(), n)

      if step % args.report_freq == 0 or step == len(train_queue)-1:
        train_loggers[i].info('train %03d %e', epoch*len(train_queue)+step, loss.item())

    if step % args.save_freq == 0 or step == len(train_queue)-1:
      for i in range(len(models)):
        torch.save(models[i].state_dict(), model_pytorch_files[i])

    if not args.validation_freq is None and step >= 400 and step % args.validation_freq == 0 and epoch == 0:
      valid_losses = infer(args, validation_queue, models, criterion, validation_loggers)
      for i in range(len(models)):
        validation_loggers[i].info(f'validation {epoch*len(train_queue)+step} {valid_losses[i]}')

  return [loss_tracker.avg for loss_tracker in loss_trackers]


def infer(args, valid_queue, models, criterion, validation_loggers):
  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.eval()

  for step, (input, target) in enumerate(valid_queue): 
    bayer_quad = Variable(input, volatile=True).float().cuda()
    target = Variable(target, volatile=True).float().cuda()
    n = bayer_quad.size(0) 

    for i, model in enumerate(models):
      pred = model(bayer_quad)
      clamped = torch.clamp(pred, min=0, max=1)
      #loss = criterion(pred[:,1,:,:], target[:,1,:,:])
      loss = criterion(clamped, target)
      loss_trackers[i].update(loss.item(), n)

  return [loss_tracker.avg for loss_tracker in loss_trackers]


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
  parser.add_argument('--momentum', type=float, default=90., help='momentum')
  parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='save frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--epochs', type=int, default=5, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  
  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')
  parser.add_argument('--training_file', type=str, default="/home/karima/cnn-data/subset_files_100k/subset_7.txt", help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, default="/home/karima/cnn-data/val_files.txt", help='filename of file with list of validation data image files')
  parser.add_argument('--test_file', type=str, default='/home/karima/cnn-data/test_files.txt', help='filename of file with list of test data image files')
  parser.add_argument('--results_file', type=str, default='training_results', help='where to store training results')
  parser.add_argument('--validation_freq', type=int, default=None, help='validation frequency')

  parser.add_argument('--multiresquad', action="store_true")
  parser.add_argument('--multiresquad2', action="store_true")
  parser.add_argument('--multiresquad3', action="store_true")

  parser.add_argument('--basicquad', action="store_true")
  parser.add_argument('--basicquadfull', action='store_true')
  parser.add_argument('--basicquadfull2', action='store_true')
  parser.add_argument('--lowres_w_fullres_f_quad', action='store_true')
  parser.add_argument('--multiresquadratio', action='store_true')

  parser.add_argument('--multiresquadfull', action="store_true")
  parser.add_argument('--multiresquadfull2', action='store_true')
  parser.add_argument('--multiresquadfull3', action='store_true')
  parser.add_argument('--multiresquadfull4', action='store_true')
  parser.add_argument('--multiresquadfull5', action='store_true')
  parser.add_argument('--multiresquadfull6', action='store_true')
  parser.add_argument('--multiresquadfull7', action='store_true')
  parser.add_argument('--multiresquadfull8', action='store_true')
  parser.add_argument('--multiresquadfull9', action='store_true')
  parser.add_argument('--multiresquadfull10', action='store_true')
  parser.add_argument('--multiresquadfull11', action='store_true')
  parser.add_argument('--multiresquadfull12', action='store_true')

  parser.add_argument('--width', type=int)
  parser.add_argument('--depth', type=int)
  parser.add_argument('--fixed_cheap', action='store_true')
  parser.add_argument('--cheap_depth', type=int)
  parser.add_argument('--chroma_width', type=int)
  parser.add_argument('--chroma_depth', type=int)
  parser.add_argument('--bias', action='store_true')
  parser.add_argument('--scale_factor', type=int)
  parser.add_argument('--full_model', action='store_true')
  parser.add_argument('--testing', action='store_true')
  
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

  if args.basicquad:
    models = [BasicQuadGreenModel(width=args.width, depth=args.depth, bias=args.bias) for i in range(args.model_initializations)]
  elif args.basicquadfull:
    models = [BasicQuadRGBModel(width=args.width, depth=args.depth, chroma_depth=args.chroma_depth, bias=args.bias) for i in range(args.model_initializations)]
  elif args.basicquadfull2:
    models = [BasicQuadRGBV2Model(width=args.width, depth=args.depth, bias=args.bias) for i in range(args.model_initializations)]
  elif args.multiresquad:
    models = [MultiresQuadGreenModel(width=args.width, depth=args.depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]
  elif args.multiresquad2:
    models = [MultiresQuadGreenV2Model(width=args.width, depth=args.depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]
  elif args.multiresquad3:
    models = [MultiresQuadGreenV3Model(width=args.width, depth=args.depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]
  elif args.multiresquadratio:
    models = [MultiresQuadGreenRatioModel(width=args.width, depth=args.depth, bias=args.bias, scale_factor=args.scale_factor, cheap_depth=args.cheap_depth, fixed_cheap=args.fixed_cheap) for i in range(args.model_initializations)]
  elif args.multiresquadfull:
    models = [MultiresQuadRGBModel(width=args.width, depth=args.depth, bias=args.bias, chroma_depth=args.chroma_depth, scale_factor=args.scale_factor) for i in range(args.model_initializations)]
  elif args.multiresquadfull2:
    models = [MultiresQuadRGBV2Model(width=args.width, depth=args.depth, chroma_width=args.chroma_width, chroma_depth=args.chroma_depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]
  elif args.multiresquadfull3:
    models = [MultiresQuadRGBV3Model(width=args.width, depth=args.depth, chroma_width=args.chroma_width, chroma_depth=args.chroma_depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]
  elif args.multiresquadfull4:
    models = [MultiresQuadRGBV4Model(width=args.width, depth=args.depth, chroma_width=args.chroma_width, chroma_depth=args.chroma_depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]
  elif args.multiresquadfull5:
    models = [MultiresQuadRGBV5Model(width=args.width, depth=args.depth, chroma_width=args.chroma_width, chroma_depth=args.chroma_depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]
  elif args.multiresquadfull6:
    models = [MultiresQuadRGBV6Model(width=args.width, depth=args.depth, chroma_width=args.chroma_width, chroma_depth=args.chroma_depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]  
  elif args.multiresquadfull7:
    models = [MultiresQuadRGBV7Model(width=args.width, depth=args.depth, chroma_width=args.chroma_width, chroma_depth=args.chroma_depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]  
  elif args.multiresquadfull8:
    models = [MultiresQuadRGBV8Model(width=args.width, depth=args.depth, chroma_width=args.chroma_width, chroma_depth=args.chroma_depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]   
  elif args.multiresquadfull9:
    models = [MultiresQuadRGBV9Model(width=args.width, depth=args.depth, chroma_width=args.chroma_width, chroma_depth=args.chroma_depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]   
  elif args.multiresquadfull10:
    models = [MultiresQuadRGBV10Model(width=args.width, depth=args.depth, chroma_width=args.chroma_width, chroma_depth=args.chroma_depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]   
  elif args.multiresquadfull11:
    models = [MultiresQuadRGBV11Model(width=args.width, depth=args.depth, chroma_width=args.chroma_width, chroma_depth=args.chroma_depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]   
  elif args.multiresquadfull12:
    models = [MultiresQuadRGBV12Model(width=args.width, depth=args.depth, chroma_width=args.chroma_width, chroma_depth=args.chroma_depth, bias=args.bias, scale_factor=args.scale_factor) for i in range(args.model_initializations)]   
  elif args.lowres_w_fullres_f_quad:
    models = [LowResSelectorFullResInterpGreenModel(width=args.width, depth=args.depth, bias=args.bias) for i in range(args.model_initializations)]

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
    
  print("parameters in model:")
  for name, param in models[0].named_parameters():
    print(f"{name} {param.size()}")
  print("-------------------")

  model_manager.save_model(models, None, model_dir)

  validation_losses, training_losses = train(args, models, 'seed', model_dir) 

  model_manager.save_model(models, None, model_dir)

  with open(args.results_file, "a+") as f:
    training_losses = [str(tl) for tl in training_losses]
    training_losses_str = ",".join(training_losses)

    validation_losses = [str(vl) for vl in validation_losses]
    validation_losses_str = ",".join(validation_losses)

    data_string = f"training losses: {training_losses_str} validation losses: {validation_losses_str}\n"

    f.write(data_string)

