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

sys.path.append(sys.path[0].split("/")[0])
sys.path.append(os.path.join(sys.path[0].split("/")[0], "train_eval_scripts"))

from dataset import FullPredictionQuadDataset, GreenQuadDataset, RGB8ChanDataset, ids_from_file, FastDataLoader
from async_loader import AsynchronousLoader

import util
from validation_variance_tracker import  VarianceTracker, LRTracker


def get_optimizers(weight_decay, lr, models):
  optimizers = [torch.optim.Adam(
      m.parameters(),
      lr=lr,
      weight_decay=weight_decay) for m in models]
  return optimizers

def create_loggers(model_dir, model_id, num_models, mode):
  log_format = '%(asctime)s %(levelname)s %(message)s'
  log_files = [os.path.join(model_dir, f'v{i}_{mode}_log') for i in range(num_models)]
  for log_file in log_files:
    if os.path.exists(log_file):
      os.remove(log_file)
  loggers = [util.create_logger(f'model_{model_id}_v{i}_{mode}_logger', logging.INFO, log_format, log_files[i]) for i in range(num_models)]
  return loggers

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

  if args.deterministic:
    sampler = torch.utils.data.sampler.SequentialSampler(train_indices)
  else:
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

  if args.deterministic:
    sampler = torch.utils.data.sampler.SequentialSampler(validation_indices)
  else:
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


def run_model(args, gpu_id, model_id, models, model_dir, experiment_logger):
  print(f"running inference for {len(models)} models on GPU {gpu_id}")

  valid_queue = create_validation_dataset(args, gpu_id)

  print(f"FINISHED creating datasets")

  validation_loggers = create_loggers(model_dir, model_id, len(models), "validation")

  models = [model.to(device=f"cuda:{gpu_id}") for model in models]
  for m in models:
    m.to_gpu(gpu_id)
    
  criterion = nn.MSELoss()

  if args.pretrained:
    for i in range(args.model_initializations):
      weight_file = os.path.join(args.model_info_dir, f"{model_id}/model_v{i}_pytorch")
      if not os.path.exists(weight_file):
        print(f"model {model_id} does not have weight files")
        return [0 for m in models]

      state_dict = torch.load(weight_file)
      models[i].load_state_dict(state_dict)
 
  valid_losses, valid_psnrs = infer(args, gpu_id, valid_queue, models, criterion)

  for i in range(len(models)):
    validation_loggers[i].info('validation %e %2.3f', valid_losses[i], valid_psnrs[i])
  return valid_psnrs


def keep_best_model(models, val_psnrs, train_loggers, validation_loggers, optimizers):
  best_index = -1
  best_psnr = -1
  
  for i, psnr in enumerate(val_psnrs):
    if psnr > best_psnr:
      best_psnr = psnr
      best_index = i

  best_model = models[best_index]
  train_logger = train_loggers[best_index]
  val_logger = validation_loggers[best_index]
  optimizer = optimizers[best_index]

  for i, m in enumerate(models):
    if i != best_index:
      m = m.cpu()
      del m
  for i, l in enumerate(train_loggers):
    if i != best_index:
      del l
  for i, l in enumerate(validation_loggers):
    if i != best_index:
      del l
  for i, o in enumerate(optimizers):
    if i != best_index:
      del o

  torch.cuda.empty_cache()
  return [best_model], [train_logger], [val_logger], [optimizer], best_index


def train_model(args, gpu_id, model_id, models, model_dir, experiment_logger, train_data=None, val_data=None):
  print(f"training {len(models)} models on GPU {gpu_id}")
  torch.cuda.set_device(gpu_id)
  cudnn.benchmark = False
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)

  models = [model.to(device=f"cuda:{gpu_id}") for model in models]
  for m in models:
    m.to_gpu(gpu_id)

  train_loggers = create_loggers(model_dir, model_id, len(models), "train")
  validation_loggers = create_loggers(model_dir, model_id, len(models), "validation")
  
  train_queue = create_train_dataset(args, gpu_id, shared_data=train_data, logger=experiment_logger)
  valid_queue = create_validation_dataset(args, gpu_id, shared_data=val_data)

  criterion = nn.MSELoss()

  if args.lrsearch:
    lr_search_start = time.time()
    lr_tracker, optimizers, chosen_lr = lr_search(args, gpu_id, models, model_id, model_dir, criterion, \
                                      train_queue, train_loggers, valid_queue, validation_loggers, experiment_logger)
    lr_search_end = time.time()
    experiment_logger.info(f"time to perform lr search: {lr_search_end - lr_search_start}")
    reuse_lr_search_epoch = lr_tracker.proposed_lrs[-1] == lr_tracker.proposed_lrs[-2]
    for lr_search_iter, variance in enumerate(lr_tracker.seen_variances):
      experiment_logger.info(f"LEARNING RATE {lr_tracker.proposed_lrs[lr_search_iter]} INDUCES VALIDATION VARIANCE {variance} AT A VALIDATION FRE0QUENCY OF {args.validation_freq}")
  else:
    reuse_lr_search_epoch = False
    chosen_lr = args.learning_rate

  experiment_logger.info(f"USING LEARNING RATE {chosen_lr}")

  if len(train_queue) > args.validation_variance_end_step:
    reuse_lr_search_epoch = False

  if not reuse_lr_search_epoch:
    cur_epoch = 0
    for m in models:
      m._initialize_parameters()
    optimizers = get_optimizers(args.weight_decay, chosen_lr, models)
  else: # we can use previous epoch's training 
    cur_epoch = 1
    val_losses, val_psnrs = infer(args, gpu_id, valid_queue, models, criterion) # compute psnrs to select which model to keep 

  for i in range(len(models)):
    train_loggers[i].info(f"STARTING TRAINING AT LR {chosen_lr}") 
    validation_loggers[i].info(f"STARTING TRAINING AT LR {chosen_lr}") 

  for epoch in range(cur_epoch, args.epochs):
    if args.keep_initializations < args.model_initializations:
      if epoch == 1:
        models, train_loggers, validation_loggers, optimizers, model_index = keep_best_model(models, val_psnrs, train_loggers, validation_loggers, optimizers)
        experiment_logger.info(f"after first epoch, validation psnrs {val_psnrs} keeping best model {model_index}")
    
    start_time = time.time()
    train_losses = train_epoch(args, gpu_id, train_queue, models, model_dir, criterion, optimizers, train_loggers, \
                              valid_queue, validation_loggers, epoch)
    end_time = time.time()
    experiment_logger.info(f"time to finish epoch {epoch} : {end_time-start_time}")
    val_losses, val_psnrs = infer(args, gpu_id, valid_queue, models, criterion)

    start_time = time.time()
    for i in range(len(models)):
      validation_loggers[i].info('validation epoch %03d %e %2.3f', epoch, val_losses[i], val_psnrs[i])
    end_time = time.time()
    experiment_logger.info(f"time to validate after epoch {epoch} : {end_time - start_time}")
  return val_psnrs


def lr_search(args, gpu_id, models, model_id, model_dir, criterion, train_queue, train_loggers, validation_queue, validation_loggers, experiment_logger):
  experiment_logger.info(f"BEGINNING LR SEARCH")
  lr_tracker = LRTracker(args.learning_rate, args.variance_max, args.variance_min)
  new_lr = args.learning_rate

  for step in range(args.lr_search_steps):
    optimizers = get_optimizers(args.weight_decay, new_lr, models)
    experiment_logger.info(f"performing LR search step {step}")
    losses, validation_variance = get_validation_variance(args, gpu_id, models, criterion, optimizers, train_queue, train_loggers, validation_queue, validation_loggers)
    new_lr = lr_tracker.update_lr(validation_variance)

    if lr_tracker.proposed_lrs[-1] == lr_tracker.proposed_lrs[-2]: 
      break
    for m in models:
      m._initialize_parameters()
        
  return lr_tracker, optimizers, new_lr


def get_validation_variance(args, gpu_id, models, criterion, optimizers, train_queue, train_loggers, validation_queue, validation_loggers):
  variance_tracker = VarianceTracker(len(models))
  loss_trackers = [util.AvgrageMeter() for m in models]
  for m in models:
    m.train()

  for step, (input, target) in enumerate(train_queue):
    if args.full_model:
      bayer, redblue_bayer, green_grgb = input 
    else:
      bayer = input
    n = bayer.size(0)

    for i, model in enumerate(models):
      model.reset()
      optimizers[i].zero_grad()
      if args.full_model:
        model_inputs = {"Input(Bayer)": bayer, 
                        "Input(Green@GrGb)": green_grgb, 
                        "Input(RedBlueBayer)": redblue_bayer}
      else:
        model_inputs = {"Input(Bayer)": bayer}

      pred = model.run(model_inputs)
      loss = criterion(pred, target)

      loss.backward()
      optimizers[i].step()
      loss_trackers[i].update(loss.item(), n)

      if step % args.report_freq == 0 or step == len(train_queue)-1:
        train_loggers[i].info('train %03d %e', step, loss.item())

    if step % args.validation_freq == 0 and step > args.validation_variance_start_step:
      valid_losses, valid_psnrs = infer(args, gpu_id, validation_queue, models, criterion)
      batchwise_valid_psnrs = [util.compute_psnr(l) for l in valid_losses]
      variance_tracker.update(batchwise_valid_psnrs)

      for i in range(len(models)):
        validation_loggers[i].info(f'validation step {step} {valid_losses[i]} {valid_psnrs[i]:2.3f}')

      if step > args.validation_variance_end_step:
        break

  return [loss_tracker.avg for loss_tracker in loss_trackers], variance_tracker.validation_variance()


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
      
      # crop
      pred = pred[..., args.crop:-args.crop, args.crop:-args.crop]

      loss = criterion(pred, target)

      loss.backward()
      optimizers[i].step()

      loss_trackers[i].update(loss.item(), n)

      if step % args.report_freq == 0 or step == len(train_queue)-1:
        train_loggers[i].info(f"train step {epoch*len(train_queue)+step} loss {loss.item():.5f}")

    if step % args.save_freq == 0 or step == len(train_queue)-1:
      if len(models) == 1: 
        model_pytorch_file = os.path.join(model_dir, "model_weights")
        torch.save(models[0].state_dict(), model_pytorch_file)
      else:
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



