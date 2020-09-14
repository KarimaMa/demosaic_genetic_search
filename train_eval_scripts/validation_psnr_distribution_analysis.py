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
import csv

sys.path.append(sys.path[0].split("/")[0])

import util
import meta_model
import model_lib
from torch_model import ast_to_model
from dataset import GreenDataset

def infer(args, valid_queue, model, criterion, validation_loggers):
  loss_tracker = util.AvgrageMeter() 
  model.eval()
  psnrs = []
  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()
    n = input.size(0)

    pred = model.run(input)
    loss = criterion(pred, target)
    psnr = util.compute_psnr(loss)
    psnrs += [psnr]

    loss_tracker.update(loss.item(), n)

  return psnrs, loss_tracker.avg 

def evaluate_model(args, model):
  validation_data = GreenDataset(data_file=args.validation_file, use_cropping=False)
  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))
  criterion = nn.MSELoss()

  validation_queue = torch.utils.data.DataLoader(
      validation_data, batch_size=1,
      sampler=torch.utils.data.SequentialSampler(validation_data),
      pin_memory=True, num_workers=0)
  
  validation_logger = util.create_logger(f'validation_logger', logging.INFO, \
                                          log_format, os.path.join(args.save, f'validation_log'))
  psnrs, avg_loss = infer(args, validation_queue, model, criterion, validation_logger)
  return psnrs, avg_loss

if __name__ == "__main__":

  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--model_path', type=str, default='models', help='path to load model')
  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')
  parser.add_argument('--model_version', type=int, help="model version to use")

  args = parser.parse_args()

  model_manager = util.ModelManager(args.model_path)
  model, ast = model_manager.load_model('', args.model_version, "gpu")
  util.create_dir(args.save)
  
  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  logger = util.create_logger('experiment_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'experiment_log'))
  logger.info("args = %s", args)

  psnrs, avg_loss = evaluate_model(args, model)
  logger.info(f"avg validation loss: {avg_loss}")

  psnr_csv = os.path.join(args.save, "validation_psnrs.csv")
  psnr_csv_f = open(psnr_csv, "w", newline="\n")
  psnr_writer = csv.writer(psnr_csv_f, delimiter=",")

  n_rows = len(psnrs)
  header = ["psnr"]

  psnr_writer.writerow(header)

  for row in range(n_rows):
    psnr_writer.writerow([psnrs[row]])





