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
sys.path.append(os.path.join(sys.path[0].split("/")[0], "train_eval_scripts"))
print(sys.path)

import util
import meta_model
import model_lib
from torch_model import ast_to_model
from dataset import GreenSharedDataset, ids_from_file
from validation_variance_tracker import  VarianceTracker, LRTracker

def get_optimizers(args, models):
  optimizers = [torch.optim.Adam(
      m.parameters(),
      lr=args.learning_rate,
      weight_decay=args.weight_decay) for m in models]
  return optimizers

def create_loggers(model_dir, model_id, num_models, mode):
  log_format = '%(asctime)s %(levelname)s %(message)s'
  log_files = [os.path.join(model_dir, f'v{i}_{mode}_log') for i in range(num_models)]
  for log_file in log_files:
    if os.path.exists(log_file):
      os.remove(log_file)
  loggers = [util.create_logger(f'model_{model_id}_v{i}_{mode}_logger', logging.INFO, log_format, log_files[i]) for i in range(num_models)]
  return loggers


def create_validation_dataset(args, inputs, inputs_shape, labels, labels_shape):
  inputs = np.frombuffer(inputs, dtype=float)
  labels = np.frombuffer(labels, dtype=float)

  inputs = inputs.reshape(inputs_shape)
  labels = labels.reshape(labels_shape)

  validation_data = GreenSharedDataset(data_file=args.validation_file, inputs=inputs, labels=labels)
  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))

  validation_queue = torch.utils.data.DataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
      pin_memory=True, num_workers=0)
  return validation_queue


def create_train_dataset(args, inputs, inputs_shape, labels, labels_shape):
  full_data_filenames = ids_from_file(args.training_file)
  used_filenames = full_data_filenames[0:int(args.train_portion)]
  inputs = np.frombuffer(inputs, dtype=float)
  labels = np.frombuffer(labels, dtype=float)
  
  inputs = inputs.reshape(inputs_shape)
  labels = labels.reshape(labels_shape)

  train_data = GreenSharedDataset(data_filenames=used_filenames, inputs=inputs, labels=labels) 

  num_train = len(train_data)
  train_indices = list(range(num_train))
  
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=True, num_workers=0)
  return train_queue


def train_model(args, gpu_id, train_queue, valid_queue, model_id, models, model_dir, experiment_logger):
  print(f"training {len(models)} models on GPU {gpu_id}")

  train_loggers = create_loggers(model_dir, model_id, len(models), "train")
  validation_loggers = create_loggers(model_dir, model_id, len(models), "validation")

  model_pytorch_files = [util.get_model_pytorch_file(model_dir, model_version) \
                        for model_version in range(len(models))]

  models = [model.to(device=f"cuda:{gpu_id}") for model in models]

  criterion = nn.MSELoss()
  optimizers = get_optimizers(args, models)

  lr_tracker, optimizers, train_loggers, validation_loggers = lr_search(args, gpu_id, models, model_id, model_dir, criterion, train_queue, train_loggers, valid_queue, validation_loggers)
  reuse_lr_search_epoch = lr_tracker.proposed_lrs[-1] == lr_tracker.proposed_lrs[-2]
  for lr_search_iter, variance in enumerate(lr_tracker.seen_variances):
    experiment_logger.info(f"PROPOSED LEARNING RATE {lr_tracker.proposed_lrs[lr_search_iter]} INDUCES VALIDATION VARIANCE {variance} AT A VALIDATION FRE0QUENCY OF {args.validation_freq}")
  experiment_logger.info(f"USING LEARNING RATE {lr_tracker.proposed_lrs[-1]}")

  if not reuse_lr_search_epoch:
    # erase logs from previous epoch and reset models
    train_loggers = create_loggers(model_dir, model_id, len(models), "train")
    validation_loggers = create_loggers(model_dir, model_id, len(models), "validation")
    cur_epoch = 0
    for m in models:
      m._initialize_parameters()
    models = [model.to(device=f"cuda:{gpu_id}") for model in models]
    optimizers = get_optimizers(args, models)
  else: # we can use previous epoch's training 
    cur_epoch = 1

  experiment_logger.info(f"learning rate stored in args {args.learning_rate}")

  for epoch in range(cur_epoch, args.epochs):
    train_losses = train_epoch(args, gpu_id, train_queue, models, criterion, optimizers, train_loggers, \
      model_pytorch_files, valid_queue, validation_loggers, epoch)
    valid_losses = infer(args, gpu_id, valid_queue, models, criterion)

    for i in range(len(models)):
      validation_loggers[i].info('validation epoch %03d %e', epoch, valid_losses[i])

  validation_psnrs = [util.compute_psnr(l) for l in valid_losses]
  train_psnrs = [util.compute_psnr(l) for l in train_losses]

  return validation_psnrs, train_psnrs


def lr_search(args, gpu_id, models, model_id, model_dir, criterion, train_queue, train_loggers, validation_queue, validation_loggers):
  lr_tracker = LRTracker(args.learning_rate, args.variance_max, args.variance_min)
  for step in range(args.lr_search_steps):
    optimizers = get_optimizers(args, models)
    losses, validation_variance = get_validation_variance(args, gpu_id, models, criterion, optimizers, train_queue, train_loggers, validation_queue, validation_loggers)
    new_lr = lr_tracker.update_lr(validation_variance)
    args.learning_rate = new_lr
    if lr_tracker.proposed_lrs[-1] == lr_tracker.proposed_lrs[-2]: 
      break
    # trying a new lr - erase logs from previous epoch and reset models
    train_loggers = create_loggers(model_dir, model_id, len(models), "train")
    validation_loggers = create_loggers(model_dir, model_id, len(models), "validation")
    for m in models:
      m._initialize_parameters()
    models = [model.to(device=f"cuda:{gpu_id}") for model in models]
        
  return lr_tracker, optimizers, train_loggers, validation_loggers


def get_validation_variance(args, gpu_id, models, criterion, optimizers, train_queue, train_loggers, validation_queue, validation_loggers):
  variance_tracker = VarianceTracker(len(models))
  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.train()

  print(f"LEN OF TRAIN QUEUE {len(train_queue)}")
  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)
    input = Variable(input, requires_grad=False).to(device=f"cuda:{gpu_id}")#cuda()
    target = Variable(target, requires_grad=False).to(device=f"cuda:{gpu_id}")#cuda()

    for i, model in enumerate(models):
      optimizers[i].zero_grad()
      pred = model.run(input)
      loss = criterion(pred, target)

      loss.backward()
      optimizers[i].step()
      loss_trackers[i].update(loss.item(), n)

      if step % args.report_freq == 0 or step == len(train_queue)-1:
        train_loggers[i].info('train %03d %e', step, loss.item())

    if step % args.validation_freq == 0 and step > 400:
      valid_losses = infer(args, gpu_id, validation_queue, models, criterion)
      valid_psnrs = [util.compute_psnr(l) for l in valid_losses]
      variance_tracker.update(valid_psnrs)
      
      for i in range(len(models)):
        validation_loggers[i].info(f'validation {step} {valid_losses[i]}')
      
  return [loss_tracker.avg for loss_tracker in loss_trackers], variance_tracker.validation_variance()


def train_epoch(args, gpu_id, train_queue, models, criterion, optimizers, train_loggers, model_pytorch_files, validation_queue, validation_loggers, epoch):
  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.train()

  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)

    input = Variable(input, requires_grad=False).to(device=f"cuda:{gpu_id}")#cuda()
    target = Variable(target, requires_grad=False).to(device=f"cuda:{gpu_id}")#cuda()

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


def infer(args, gpu_id, valid_queue, models, criterion):
  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, requires_grad=False).to(device=f"cuda:{gpu_id}")#cuda()
      target = Variable(target, requires_grad=False).to(device=f"cuda:{gpu_id}")#cuda()
      n = input.size(0)

      for i, model in enumerate(models):
        pred = model.run(input)
        loss = criterion(pred, target)
        loss_trackers[i].update(loss.item(), n)

  return [loss_tracker.avg for loss_tracker in loss_trackers]
