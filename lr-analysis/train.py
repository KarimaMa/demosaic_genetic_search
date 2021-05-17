import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import logging
import argparse
import os
import sys
import time
import csv

sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "train_eval_scripts"))

from dataset import FullPredictionQuadDataset, GreenQuadDataset, RGB8ChanDataset, ids_from_file, FastDataLoader
from async_loader import AsynchronousLoader

import util


def get_optimizers(weight_decay, lr, models):
  optimizers = [torch.optim.Adam(
      m.parameters(),
      lr=lr,
      weight_decay=weight_decay) for m in models]
  return optimizers


"""
write training performance ot csv files
"""
class SampleLogger():
  def __init__(self, model_dir, init, lr, mode, fieldnames):
    self.filename = os.path.join(model_dir, f'v{init}_lr{lr}_{mode}.csv')
    self.file = open(self.filename, 'w')
    self.fieldnames = fieldnames
    self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
    self.writer.writeheader()

  def write(self, data):
    self.writer.writerow(data)
    self.file.flush()

  def close(self):
    self.file.close()


def create_train_dataset(args, gpu_id, shared_data=None, logger=None, batch_size=None):
  device = torch.device(f'cuda:{gpu_id}')

  if args.full_model:
    train_data = FullPredictionQuadDataset(data_file=args.training_file, RAM=args.ram, lazyRAM=args.lazyram, \
                                          shared_data=shared_data, logger=logger)
  elif args.rgb8chan:
    train_data = RGB8ChanDataset(data_file=args.training_file)
  else:
    train_data = GreenQuadDataset(data_file=args.training_file) 

  num_train = len(train_data)
  train_indices = list(range(num_train))

  sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)

  logger.info("Creating FastDataLoader...")
  if not batch_size is None:
    bs = batch_size
  else:
    bs = args.batch_size

  train_queue = FastDataLoader(
      train_data, batch_size=bs,
      sampler=sampler,
      pin_memory=True, num_workers=8)
  logger.info("Finished creating FastDataLoader")

  logger.info("Creating AsyncLoader...")
  loader = AsynchronousLoader(train_queue, device)
  logger.info("Finished creating AsyncLoader")
  return loader


def create_validation_dataset(args, gpu_id, shared_data=None, batch_size=None):
  device = torch.device(f'cuda:{gpu_id}')

  if args.full_model:
    validation_data = FullPredictionQuadDataset(data_file=args.validation_file, RAM=args.ram, lazyRAM=args.lazyram,\
                                               shared_data=shared_data)
  elif args.rgb8chan:
    validation_data = RGB8ChanDataset(data_file=args.validation_file)
  else:
    validation_data = GreenQuadDataset(data_file=args.validation_file)

  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))

  sampler = torch.utils.data.sampler.SubsetRandomSampler(validation_indices)

  if not batch_size is None:
    bs = batch_size
  else:
    bs = args.batch_size

  validation_queue = FastDataLoader(
      validation_data, batch_size=bs,
      sampler=sampler,
      pin_memory=True, num_workers=8)

  loader = AsynchronousLoader(validation_queue, device)

  return loader


def train_model(args, gpu_id, model_id, models, model_dir, experiment_logger, train_data=None, val_data=None):
  print(f"training model {model_id} on GPU {gpu_id} lr {args.learning_rate}")
  torch.cuda.set_device(gpu_id)
  cudnn.benchmark = False
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)

  models = [model.to(device=f"cuda:{gpu_id}") for model in models]
  for m in models:
    m.to_gpu(gpu_id)

  train_loggers = [SampleLogger(model_dir, init, args.learning_rate, "train", ["step", "loss"]) for init in range(args.model_initializations)]
  val_sample_loggers = [SampleLogger(model_dir, init, args.learning_rate, "val-sample", ["step", "psnr"]) for init in range(args.model_initializations)]
  val_epoch_loggers = [SampleLogger(model_dir, init, args.learning_rate, "val", ["epoch", "psnr"]) for init in range(args.model_initializations)]

  train_queue = create_train_dataset(args, gpu_id, shared_data=train_data, logger=experiment_logger)
  valid_queue = create_validation_dataset(args, gpu_id, shared_data=val_data)

  criterion = nn.MSELoss()

  experiment_logger.info(f"USING LEARNING RATE {args.learning_rate}")

  for m in models:
    m._initialize_parameters()
  optimizers = get_optimizers(args.weight_decay, args.learning_rate, models)

  for epoch in range(args.epochs):
    start_time = time.time()
    train_losses = train_epoch(args, gpu_id, train_queue, models, model_dir, criterion, optimizers, train_loggers, \
                              valid_queue, val_sample_loggers, epoch)
    end_time = time.time()
    experiment_logger.info(f"time to finish epoch {epoch} : {end_time-start_time}")
    val_losses, val_psnrs = infer(args, gpu_id, valid_queue, models, criterion)

    start_time = time.time()
    for i in range(len(models)):
      val_epoch_loggers[i].write({"epoch": epoch, "psnr": val_psnrs[i]})
    end_time = time.time()
    experiment_logger.info(f"time to validate after epoch {epoch} : {end_time - start_time}")

  for logger in train_loggers:
    logger.close()
  for logger in val_sample_loggers:
    logger.close()
  for logger in val_epoch_loggers:
    logger.close()

  return val_psnrs


def sample_validation_psnr(epoch, step, args):
  return (epoch == 0) and (step % args.validation_freq == 0) and (step > args.validation_variance_start_step) and (step < args.validation_variance_end_step)


def train_epoch(args, gpu_id, train_queue, models, model_dir, criterion, optimizers, train_loggers, validation_queue, validation_loggers, epoch):
  loss_trackers = [util.AvgrageMeter() for m in models]

  for m in models:
    m.train()

  for step, (input, target) in enumerate(train_queue):
    if args.full_model:
      bayer, redblue_bayer, green_grgb = input 
    else:
      bayer = input
    target = target[..., args.crop:-args.crop, args.crop:-args.crop]

    n = bayer.size(0)

    for i, model in enumerate(models):
      optimizers[i].zero_grad()
      model.reset()
      if args.full_model:
        model_inputs = {"Input(Bayer)": bayer, 
                        "Input(Green@GrGb)": green_grgb, 
                        "Input(RedBlueBayer)": redblue_bayer}
      else:
        model_inputs = {"Input(Bayer)": bayer}

      pred = model.run(model_inputs) 
      pred = pred[..., args.crop:-args.crop, args.crop:-args.crop]

      loss = criterion(pred, target)
      loss.backward()
      optimizers[i].step()
      loss_trackers[i].update(loss.item(), n)

      if step % args.report_freq == 0 or step == len(train_queue)-1:
        train_loggers[i].write({"step": epoch * len(train_queue)+step, "loss": loss.item()})

    if sample_validation_psnr(epoch, step, args): 
      valid_losses, val_psnrs = infer(args, gpu_id, validation_queue, models, criterion)

      for i in range(len(models)):
        validation_loggers[i].write({"step": epoch * len(train_queue)+step, "psnr": val_psnrs[i]})

    if step % args.save_freq == 0 or step == len(train_queue)-1:
      model_pytorch_files = [util.get_model_pytorch_file(model_dir, model_version, epoch) for model_version in range(len(models))]
      for i in range(len(models)):
        torch.save(models[i].state_dict(), model_pytorch_files[i])

  return [loss_tracker.avg for loss_tracker in loss_trackers] #, [psnr_tracker.avg for psnr_tracker in psnr_trackers]


def infer(args, gpu_id, valid_queue, models, criterion):
  loss_trackers = [util.AvgrageMeter() for m in models]
  psnr_trackers = [util.AvgrageMeter() for m in models]

  for m in models:
    m.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      if args.full_model:
        bayer, redblue_bayer, green_grgb = input 
      else:
        bayer = input

      target = target[..., args.crop:-args.crop, args.crop:-args.crop]

      n = bayer.size(0)

      for i, model in enumerate(models):
        model.reset()
        if args.full_model:
          model_inputs = {"Input(Bayer)": bayer, 
                          "Input(Green@GrGb)": green_grgb, 
                          "Input(RedBlueBayer)": redblue_bayer}
          pred = model.run(model_inputs)
        else:
          model_inputs = {"Input(Bayer)": bayer}
       
        pred = model.run(model_inputs)

        # crop
        pred = pred[..., args.crop:-args.crop, args.crop:-args.crop]
        
        clamped = torch.clamp(pred, min=0, max=1)

        loss = criterion(clamped, target)
        loss_trackers[i].update(loss.item(), n)

        # compute running psnr
        per_image_mse = (clamped-target).square().mean(-1).mean(-1).mean(-1)
        per_image_psnr = -10.0*torch.log10(per_image_mse)
        batch_avg_psnr = per_image_psnr.sum(0) / n
        psnr_trackers[i].update(batch_avg_psnr.item(), n)

  return [loss_tracker.avg for loss_tracker in loss_trackers], [psnr_tracker.avg for psnr_tracker in psnr_trackers]



