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
import math
sys.path.append(sys.path[0].split("/")[0])

import util
import meta_model
import model_lib
from torch_model import ast_to_model
from dataset import GreenDataset


def compute_psnr(loss):
  return 10*math.log(math.pow(255,2) / math.pow(math.sqrt(loss)*255, 2),10)

def detect_divergence(psnrs, threshold):
  return (max(psnrs) - min(psnrs)) > threshold


def find_lr(args, model_ast, model_dir):

  validation_loggers = [util.create_logger(f'v{i}_validation_logger', logging.INFO, \
                                          log_format, os.path.join(model_dir, f'v{i}_validation_log')) \
                      for i in range(args.model_initializations)]

  train_data = GreenDataset(args.training_file, use_cropping=False) 
  validation_data = GreenDataset(args.validation_file, use_cropping=False)

  num_train = len(train_data)
  train_indices = list(range(int(num_train*args.train_portion)))
  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SequentialSampler(train_data),
      pin_memory=True, num_workers=0)

  validation_queue = torch.utils.data.DataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
      pin_memory=True, num_workers=0)
  
  criterion = nn.MSELoss()

  LR = args.max_learning_rate
  BEST_PSNR = -1.0*float('inf')
  PREV_LR = None

  while LR > args.min_lr:

    models = [model_ast.ast_to_model().cuda() for i in range(args.model_initializations)]
    for m in models:
      m._initialize_parameters()

    models = [m.cuda() for m in models]

    optimizers = [torch.optim.Adam(
      m.parameters(), LR,
      weight_decay=args.weight_decay) for m in models]

    lr_log_dir = os.path.join(model_dir, f"LR_{LR}")
    util.create_dir(lr_log_dir)

    train_losses = train(args, models, train_queue, lr_log_dir, optimizers, criterion)
    train_psnrs = [compute_psnr(l) for l in train_losses]

    # test partially trained model on validation data
    valid_losses = infer(args, validation_queue, models, criterion)
    valid_psnrs = [compute_psnr(l) for l in valid_losses]

    for i in range(len(models)):
      validation_loggers[i].info(f'{model_dir} v{i} validation LR {LR} loss {valid_losses[i]}')

    PSNR = max(valid_psnrs)
    print(f"LR {LR} train_psnrs {train_psnrs}  valid_psnrs {valid_psnrs}")
    models_diverge = detect_divergence(valid_psnrs, args.divergence_threshold)

    if models_diverge:
      PREV_LR = LR/2 # DIVERGENT LR MUST HALVE REGARDLESS OF SMALLER LR'S PSNR
      LR /= 2
      print(f"{model_dir} DIVERGE {max(valid_psnrs) - min(valid_psnrs)}")
    elif PSNR > (BEST_PSNR + args.eps):
      PREV_LR = LR
      LR /= 2
    else:
      return PREV_LR

    BEST_PSNR = max(BEST_PSNR, PSNR)

  return args.min_lr


"""
Trains models for a predetermined number of steps
"""
def train(args, models, train_queue, model_dir, optimizers, criterion):
  for m in models:
    m.train()
  loss_trackers = [util.AvgrageMeter() for m in models]
  log_format = '%(asctime)s %(levelname)s %(message)s'
  train_loggers = [util.create_logger(f'v{i}_train_logger', logging.INFO, \
                                      log_format, os.path.join(model_dir, f'v{i}_train_log'))\
                  for i in range(len(models))]

  model_pytorch_files = [util.get_model_pytorch_file(model_dir, model_version) \
                        for model_version in range(len(models))]

  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    losses = []
    for i, model in enumerate(models):
      optimizers[i].zero_grad()
      pred = model.run(input)
      loss = criterion(pred, target)

      loss.backward()
      optimizers[i].step()

      loss_trackers[i].update(loss.item(), n)
      losses.append(loss.item())
      # log every batch loss
      train_loggers[i].info('train %03d %e', step*args.batch_size, loss.item())

    if step % args.save_freq == 0:
      for i in range(len(models)):
        torch.save(models[i].state_dict(), model_pytorch_files[i])

    if step == args.decision_point:
      break
  #return losses
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
  parser.add_argument('--max_learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  parser.add_argument('--save_freq', type=float, default=2000, help='save frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--decision_point', type=int, default=200, help='number of batches to run before deciding whether to halve learning rate')
  parser.add_argument('--eps', type=float, help='minimum MSE improvement required to halve learning rate')
  parser.add_argument('--min_lr', type=float, help='minimum allowed learning rate')
  parser.add_argument('--divergence_threshold', type=float, help='MSE difference threshold between model initializations to detect divergence')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--multires_model', action='store_true')
  parser.add_argument('--demosaicnet', action='store_true')
  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--subset_id', type=int, help='training subset id to use') 
  parser.add_argument('--training_file', type=str, help='file with list of training subset files') 
  parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')
  parser.add_argument('--results_file', type=str, default='chosen_lr_results', help='where to store training results')

  args = parser.parse_args()

  args.model_path = os.path.join(args.save, args.model_path)
  args.results_file = os.path.join(args.save, args.results_file)
  model_manager = util.ModelManager(args.model_path)

  model_dir = model_manager.model_dir(f'subset_{args.subset_id}')
  util.create_dir(model_dir)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  logger = util.create_logger('training_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'training_log'))
  logger.info("args = %s", args)

  if args.multires_model:
    logger.info(f"FINDING LR FOR MULTIRES GREEN ON SUBSET {args.subset_id}")
    green = model_lib.multires_green_model()
  elif args.demosaicnet:
    logger.info(f"FINDING LR FOR DEMOSAICNET GREEN ON SUBSET {args.subset_id}")
    green = model_lib.mini_demosaicnet()
  else:
    logger.info(f"FINDING LR FOR BASIC GREEN ON SUBSET {args.subset_id}")
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

  chosen_lr = find_lr(args, green, model_dir)
  logger.info(f"LR for subset {args.subset_id} {chosen_lr}")

  with open(args.results_file, "a+") as f:
    f.write(f"LR for subset {args.subset_id} {chosen_lr}\n")


