import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import logging
import argparse
import glob
import os
import sys
import random 
import numpy as np

sys.path.append(sys.path[0].split("/")[0])

import util
import meta_model
import model_lib
from torch_model import ast_to_model
from dataset import GreenDataset, ids_from_file
from train_seed_model import train_epoch


def train(args, models, model_id, model_dir):
  print(f"training {len(models)} models")
  log_format = '%(asctime)s %(levelname)s %(message)s'
  train_loggers = [util.create_logger(f'{model_id}_v{i}_train_logger', logging.INFO, \
                                      log_format, os.path.join(model_dir, f'v{i}_train_log'))\
                  for i in range(len(models))]

  validation_loggers = [util.create_logger(f'{model_id}_v{i}_validation_logger', logging.INFO, \
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

  full_data_filenames = ids_from_file(args.training_subset)
  data_filenames = full_data_filenames[0:args.num_images]

  num_train = args.num_images - args.kfold_size
  num_validation = args.kfold_size

  kfold_start = args.kfold_id*args.kfold_size
  kfold_end = kfold_start + args.kfold_size
  validation_filenames = data_filenames[kfold_start:kfold_end]

  data_files_set = set(data_filenames)
  validation_files_set = set(validation_filenames)
  train_filenames = list(data_files_set - validation_files_set)

  train_data = GreenDataset(data_filenames=train_filenames, use_cropping=args.use_cropping) 
  validation_data = GreenDataset(data_filenames=validation_filenames, use_cropping=args.use_cropping)

  train_indices = list(range(num_train))
  validation_indices = list(range(num_validation))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=True, num_workers=0)

  validation_queue = torch.utils.data.DataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
      pin_memory=True, num_workers=0)

  print(f"kfold train num batches: {len(train_queue)}")
  print(f"kfold validation num batches: {len(validation_queue)}")

  for epoch in range(args.epochs):
    # training
    train_losses = train_epoch(args, train_queue, models, criterion, optimizers, train_loggers, \
      model_pytorch_files, validation_queue, validation_loggers, epoch)
    # validation
    valid_losses = infer(args, validation_queue, models, criterion, validation_loggers)
    for i in range(len(models)):
      validation_loggers[i].info('validation epoch %03d %e', epoch, valid_losses[i])

  return valid_losses, train_losses


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, help='init learning rate')
  parser.add_argument('--weight_decay', type=float, help='weight decay')
  parser.add_argument('--report_freq', type=int, default=200, help='report frequency')
  parser.add_argument('--save_freq', type=int, default=500, help='save frequency')
  parser.add_argument('--validation_freq', type=int, default=None, help='validation frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--epochs', type=int, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--subset_id', type=int, help='training subset id to use')  
  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--save', type=str, default='RAND_DATA_SUBSET_TRAIN_MULTIRES', help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--kfold_size', type=float, help='portion of data used for validation')
  parser.add_argument('--num_images', type=float, help='data set size')
  parser.add_argument('--training_subset', type=str, help='file with list of training subset files')
  parser.add_argument('--results_file', type=str, default='training_results', help='where to store training results')
  parser.add_argument('--multires_model', action='store_true')
  parser.add_argument('--multires_model2d', action='store_true')
  parser.add_argument('--basic_model2d', action='store_true')
  parser.add_argument('--demosaicnet', action='store_true')
  parser.add_argument('--ahd1d', action='store_true')
  parser.add_argument('--ahd2d', action='store_true')
  parser.add_argument('--use_cropping', action='store_true')
  parser.add_argument('--kfold_id', type=int, help="which kfold to use")

  args = parser.parse_args()
  util.create_dir(args.save, scripts_to_save=glob.glob('*.py'))
  args.results_file = os.path.join(args.save, args.results_file)
  args.model_path = os.path.join(args.save, args.model_path)
  args.kfold_size = int(args.kfold_size)
  args.num_images = int(args.num_images)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  experiment_logger = util.create_logger('experiment_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'experiment_log'))
  experiment_logger.info("args = %s", args)

  model_manager = util.ModelManager(args.model_path)

  if args.multires_model:
    experiment_logger.info("TRAINING MULTIRES GREEN")
    green = model_lib.multires_green_model()
  elif args.multires_model2d:
    experiment_logger.info("TRAINING MULTIRES2D GREEN")
    green = model_lib.multires2D_green_model()
  elif args.demosaicnet:
    experiment_logger.info("TRAINING DEMOSAICNET GREEN")
    green = model_lib.mini_demosaicnet()
  elif args.ahd1d:
    experiment_logger.info("TRAINING AHD1D GREEN")
    green = model_lib.ahd1D_green_model()
  elif args.ahd2d:
    experiment_logger.info("TRAINING AHD2D GREEN")
    green = model_lib.ahd2D_green_model()
  elif args.basic_model2d:
    experiment_logger.info("TRAINING BASIC2D GREEN")
    green = model_lib.basic2D_green_model()
  else:    
    experiment_logger.info("TRAINING BASIC GREEN")
    full_model = meta_model.MetaModel()
    full_model.build_default_model() 
    green = full_model.green

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = False
  torch.manual_seed(args.seed)

  cudnn.enabled=True
  cudnn.deterministic=True
  torch.cuda.manual_seed(args.seed)

  with open(args.training_subset, "r") as f:
    training_subsets = [l.strip() for l in f]

  print(f"training models on subset {args.subset_id}")

  model_dir = model_manager.model_dir(f'subset_{args.subset_id}')
  util.create_dir(model_dir)

  if not torch.cuda.is_available():
    sys.exit(1)

  models = [green.ast_to_model().cuda() for i in range(args.model_initializations)]

  for m in models:
    m._initialize_parameters()

  validation_losses, training_losses = train(args, models, f'subset_{args.subset_id}', model_dir) 
  model_manager.save_model(models, green, model_dir)

  with open(args.results_file, "a+") as f:
    training_losses = [str(tl) for tl in training_losses]
    training_losses_str = ",".join(training_losses)

    validation_losses = [str(vl) for vl in validation_losses]
    validation_losses_str = ",".join(validation_losses)

    data_string = f"subset {args.subset_id} training losses: {training_losses_str} validation losses: {validation_losses_str}\n"

    f.write(data_string)

