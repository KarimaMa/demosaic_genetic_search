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
from dataset import GreenDataset, Dataset, FastDataLoader
from dataset import GreenQuadDataset, QuadDataset, FastDataLoader
from tree import print_parents


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
  for m in models:
    m.to_gpu(args.gpu)
    
  criterion = nn.MSELoss()

  optimizers = [torch.optim.Adam(
      m.parameters(),
      args.learning_rate) for m in models]

  if not args.multiresquadgreen and not args.demosaicnet:
    if args.full_model:
      train_data = Dataset(data_file=args.training_file,
                        green_input=args.use_green_input, 
                        green_pred_input=args.use_green_pred,
                        green_file=args.green_training_file, 
                        green_output=(not args.full_model))

      validation_data = Dataset(data_file=args.validation_file, 
                              green_input=args.use_green_input, 
                              green_pred_input=args.use_green_pred,
                              green_file=args.green_validation_file,
                              green_output=(not args.full_model))

    else:
      train_data = GreenDataset(data_file=args.training_file)
      validation_data = GreenDataset(data_file=args.validation_file)
      test_data = GreenDataset(data_file=args.test_file)

  else:
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
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
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
    if args.use_green_input:
      bayer, green = input 
      green_input = Variable(green, requires_grad=False).cuda()
    else:
      bayer = input

    bayer_input = Variable(bayer, requires_grad=False).float().cuda()
    target = Variable(target, requires_grad=False).float().cuda()

    n = bayer.size(0)

    for i, model in enumerate(models):
      optimizers[i].zero_grad()
      if args.use_green_input:
        model_inputs = {"Input(Bayer)": bayer_input, "Input(Green)": green_input}
        pred = model.run(model_inputs)
      else:
        model_inputs = {"Input(Bayer)": bayer_input}
        pred = model.run(model_inputs)

      if args.testing:
        print("pred")
        print(pred[0,:,0:4,0:4])
        print("target")
        print(target[0,:,0:4,0:4])
        exit()

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
    if args.use_green_input:
      bayer, green = input 
      green_input = Variable(green, volatile=True).cuda()
    else:
      bayer = input

    bayer_input = Variable(bayer, volatile=True).float().cuda()
    target = Variable(target, volatile=True).float().cuda()

    n = bayer.size(0)

    for i, model in enumerate(models):
      if args.use_green_input:
        model_inputs = {"Input(Bayer)": bayer_input, "Input(Green)": green_input}
        pred = model.run(model_inputs)
      else:
        model_inputs = {"Input(Bayer)": bayer_input}
        pred = model.run(model_inputs)
 
      loss = criterion(pred, target)
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
  parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')

  parser.add_argument('--multires_model', action='store_true')
  parser.add_argument('--fast_multires_model', action='store_true')
  parser.add_argument('--demosaicnet', action='store_true')
  parser.add_argument('--ahd', action='store_true')
  parser.add_argument('--ahd2d', action='store_true')
  parser.add_argument('--basic_model2d', action='store_true')
  parser.add_argument('--chroma', action='store_true')
  parser.add_argument('--chroma_no_sub', action='store_true')
  parser.add_argument('--chroma_no_extraction', action='store_true')
  parser.add_argument('--multiresquadgreen', action='store_true')

  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')
  parser.add_argument('--training_file', type=str, default="/home/karima/cnn-data/subset_files_100k/subset_7.txt", help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, default="/home/karima/cnn-data/val_files.txt", help='filename of file with list of validation data image files')
  parser.add_argument('--test_file', type=str, default="/home/karima/cnn-data/test_files.txt")

  parser.add_argument('--results_file', type=str, default='training_results', help='where to store training results')
  parser.add_argument('--validation_freq', type=int, default=None, help='validation frequency')

  parser.add_argument('--use_green_input', action="store_true")
  parser.add_argument('--full_model', action="store_true")
  parser.add_argument('--use_green_pred', action="store_true", help="whether to use precomputed green predictions")
  parser.add_argument('--green_training_file', type=str, help="filename of file with list of precomputed green for training data")
  parser.add_argument('--green_validation_file', type=str, help="filename of file with list of precomputed green for validation data")

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

  if args.multires_model:
    logger.info("TRAINING MULTIRES GREEN")
    model = model_lib.multires_green_model()
  elif args.fast_multires_model:
    logger.info("TRAINING FAST MULTIRES GREEN")
    model = model_lib.fast_multires_green_model()
  elif args.demosaicnet:
    logger.info("TRAINING DEMOSAICNET GREEN")
    model = model_lib.GreenDemosaicknet(3, 8)
  elif args.ahd:
    logger.info(f"TRAINING AHD GREEN")
    model = model_lib.ahd1D_green_model()
  elif args.ahd2d:
    logger.info(f"TRAINING AHD2D GREEN")
    model = model_lib.ahd2D_green_model()
  elif args.basic_model2d:
    logger.info(f"TRAINING BASIC_MODEL2D GREEN")
    model = model_lib.basic2D_green_model()
  elif args.chroma:
    logger.info(f"TRAINING CHROMA MODEL USING GREEN INPUT")
    model = model_lib.simple_full_model_green_input()
  elif args.chroma_no_sub:
    logger.info(f"TRAINING NO SUB CHROMA MODEL USING GREEN INPUT")
    model = model_lib.no_sub_full_model_green_input()
  elif args.chroma_no_extraction:
    logger.info(f"TRAINING CHROMA WITHOUT EXTRACTOR")
    model = model_lib.simple_chroma_no_extraction()
  elif args.multiresquadgreen:
    logger.info("TRAINING MULTIRES QUAD GREEN")
    model = model_lib.MultiresQuadGreenModel(2, 10)
  else:
    logger.info("TRAINING BASIC GREEN")
    model = model_lib.basic1D_green_model()

  print_parents(model)

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

  torch_models = [model.ast_to_model().cuda() for i in range(args.model_initializations)]
  for m in torch_models:
    m._initialize_parameters()
    
  for name, param in torch_models[0].named_parameters():
    print(f"{name} {param.size()}")

  model_manager.save_model(torch_models, model, model_dir)

  validation_losses, training_losses = train(args, torch_models, 'seed', model_dir) 

  model_manager.save_model(torch_models, model, model_dir)

  reloaded = model_manager.load_model_ast('seed')
  preorder_nodes = reloaded.preorder()
  print("In tree nodes with multiple parents")
  for n in preorder_nodes:
    if type(n.parent) is tuple:
      print(f"node {id(n)} parent {n.parent}{n.dump()}")
  print("------------")

  print("partners in new tree:")
  for n in preorder_nodes:
    if hasattr(n, "partner_set"):
      print(f"node {n.name} with id {id(n)} partners:")
      for p, pid in n.partner_set:
        print(f"{p.name} {id(p)}")
      print("-------")
  print("========")


  with open(args.results_file, "a+") as f:
    training_losses = [str(tl) for tl in training_losses]
    training_losses_str = ",".join(training_losses)

    validation_losses = [str(vl) for vl in validation_losses]
    validation_losses_str = ",".join(validation_losses)

    data_string = f"training losses: {training_losses_str} validation losses: {validation_losses_str}\n"

    f.write(data_string)


