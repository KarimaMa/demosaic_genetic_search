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

import cost
import util
import superres_model_lib
from torch_model import ast_to_model
from superres_dataset import GreenQuadDataset
from dataset import FastDataLoader
from tree import print_parents
import demosaic_ast


def run(args, models, model_id, model_dir):
  print(f"training {len(models)} models")
  log_format = '%(asctime)s %(levelname)s %(message)s'

  if args.train:
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
 
  if args.full_model:
      raise NotImplementedError
  else:
    if args.train:
         train_data = GreenQuadDataset(data_file=args.training_file) 
    validation_data = GreenQuadDataset(data_file=args.validation_file) 
    test_data = GreenQuadDataset(data_file=args.test_file) 

    #   train_data = GreenQuadDataset(input_data_file=args.train_input, target_data_file=args.train_target) 
    # validation_data = GreenQuadDataset(input_data_file=args.val_input, target_data_file=args.val_target) 
    # test_data = GreenQuadDataset(input_data_file=args.test_input, target_data_file=args.test_target) 
    
  if args.train:
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
 
  if args.train:
    for epoch in range(args.epochs):
      train_losses = train_epoch(args, train_queue, models, criterion, optimizers, train_loggers, \
        model_pytorch_files, validation_queue, validation_loggers, epoch)
      print(f"finished epoch {epoch}")
      valid_losses, val_psnrs = infer(args, validation_queue, models, criterion, validation_loggers)
      test_losses, test_psnrs = infer(args, test_queue, models, criterion, test_loggers)

      for i in range(len(models)):
        validation_loggers[i].info('validation epoch %03d mse %e psnr %e', epoch, valid_losses[i], val_psnrs[i])
        test_loggers[i].info('test epoch %03d mse %e psnr %e', epoch, test_losses[i], test_psnrs[i])

    return valid_losses, train_losses
  else:
    valid_losses, val_psnrs = infer(args, validation_queue, models, criterion, validation_loggers)
    test_losses, test_psnrs = infer(args, test_queue, models, criterion, test_loggers)

    for i in range(len(models)):
      print(f"valid loss {valid_losses[i]}")
      validation_loggers[i].info('validation mse %e psnr %e', valid_losses[i], val_psnrs[i])
      test_loggers[i].info('test mse %e psnr %e', test_losses[i], test_psnrs[i])

    return valid_losses, test_losses 



def train_epoch(args, train_queue, models, criterion, optimizers, train_loggers, model_pytorch_files, validation_queue, validation_loggers, epoch):
  loss_trackers = [util.AvgrageMeter() for m in models]
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

  for m in models:
    m.train()

  for step, (input, target) in enumerate(train_queue):
    if args.full_model:
      if args.xtrans:
        raise NotImplementedError
      else:
        bayer, redblue_bayer, green_grgb = input 
        bayer = bayer.cuda()
        redblue_bayer = redblue_bayer.cuda()
        green_grgb = green_grgb.cuda()

    else:
      mosaic = input
      mosaic = mosaic.cuda()
      target = target.cuda()

    if args.testing:
      print("target")
      print(target[0,:,6:12,6:12])
      print("mosaic")
      print(mosaic[0,:,1,1])

    if args.crop > 0:
      target = target[..., args.crop:-args.crop, args.crop:-args.crop]

    n = target.size(0)

    for i, model in enumerate(models):
      optimizers[i].zero_grad()
      model.reset()
      if args.full_model:
        model_inputs = {"Input(Bayer)": bayer, 
                        "Input(Green@GrGb)": green_grgb, 
                        "Input(RedBlueBayer)": redblue_bayer}
        pred = model.run(model_inputs)
      else:
        model_inputs = {"Input(Mosaic)": mosaic}
        pred = model.run(model_inputs)

      if args.crop > 0:
        pred = pred[..., args.crop:-args.crop, args.crop:-args.crop]

      if args.testing:
        print("pred")
        print(f"{pred[0,:,6:12,6:12].cpu().detach().numpy()}")
        diff = pred - target
        print("diff")
        print(diff[0,:,6:12,6:12])
        exit()

      loss = criterion(pred, target) 
      loss.backward()
      optimizers[i].step()
      loss_trackers[i].update(loss.item(), n)

      if step % args.report_freq == 0 or step == len(train_queue)-1:
        # compute batch average psnr
        clamped = torch.clamp(pred, min=0, max=1).detach()
        per_image_mse = (clamped-target).square().mean(-1).mean(-1).mean(-1)
        per_image_psnr = -10.0*torch.log10(per_image_mse)
        batch_avg_psnr = per_image_psnr.sum(0) / n

        train_loggers[i].info(f'train step {epoch*len(train_queue)+step} loss {loss.item():.5f} psnr {batch_avg_psnr:.3f}')

    if step % args.save_freq == 0 or step == len(train_queue)-1:
      for i in range(len(models)):
        torch.save(models[i].state_dict(), model_pytorch_files[i])

    if not args.validation_freq is None and step >= 400 and step % args.validation_freq == 0 and epoch == 0:
      valid_losses, val_psnrs = infer(args, validation_queue, models, criterion, validation_loggers)
      for i in range(len(models)):
        validation_loggers[i].info(f'validation epoch {epoch*len(train_queue)+step} mse {valid_losses[i]} psnr {val_psnrs[i]}')

  return [loss_tracker.avg for loss_tracker in loss_trackers]


def infer(args, valid_queue, models, criterion, validation_loggers):
  loss_trackers = [util.AvgrageMeter() for m in models]
  psnr_trackers = [util.AvgrageMeter() for m in models]

  for m in models:
    m.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      if args.full_model:
        bayer, redblue_bayer, green_grgb = input 
        bayer = bayer.cuda()
        redblue_bayer = redblue_bayer.cuda()
        green_grgb = green_grgb.cuda()
      else:
        mosaic = input
        mosaic = mosaic.cuda()
        target = target.cuda()
      
      if args.crop > 0:
        target = target[..., args.crop:-args.crop, args.crop:-args.crop]

      n = target.size(0)

      for i, model in enumerate(models):
        model.reset()
        if args.full_model:
          model_inputs = {"Input(Bayer)": bayer, 
                          "Input(Green@GrGb)": green_grgb, 
                          "Input(RedBlueBayer)": redblue_bayer}
          pred = model.run(model_inputs)
        else:
          model_inputs = {"Input(Mosaic)": mosaic}
          pred = model.run(model_inputs)

        clamped = torch.clamp(pred, min=0, max=1)

        if args.crop > 0:
          clamped = clamped[..., args.crop:-args.crop, args.crop:-args.crop]
          
        loss = criterion(clamped, target)
        loss_trackers[i].update(loss.item(), n)

        # compute running psnr
        per_image_mse = (clamped-target).square().mean(-1).mean(-1).mean(-1)
        per_image_psnr = -10.0*torch.log10(per_image_mse)
        batch_avg_psnr = per_image_psnr.sum(0) / n

        # average per-image psnr
        psnr_trackers[i].update(batch_avg_psnr.item(), n)

  return [loss_tracker.avg for loss_tracker in loss_trackers], [psnr_tracker.avg for psnr_tracker in psnr_trackers]


def set_no_grad(model, no_grad):
  nodes = model.preorder()
  for n in nodes:
    if type(n) is demosaic_ast.Input:
      if n.name == "Input(GreenExtractor)":
        n.no_grad = no_grad            
      elif hasattr(n, "node"): # other input ops my run submodels that also
        n.no_grad = no_grad
        set_no_grad(n.node, no_grad)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
  parser.add_argument('--momentum', type=float, default=90., help='momentum')
  parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
  parser.add_argument('--save_freq', type=float, default=5000, help='save frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')

  parser.add_argument('--green_demosaicnet', action='store_true')
  parser.add_argument('--multiresquadgreen', action='store_true')
 
  parser.add_argument('--green_model_ast', type=str, help="green model ast to use for chroma model")
  parser.add_argument('--no_grad', action='store_true', help='whether or not to backprop through green model')
  parser.add_argument('--green_model_id', type=int, help='id of green model used')

  parser.add_argument('--depth', type=int, help='num conv layers in model')
  parser.add_argument('--width', type=str, help='num channels in model layers')
  parser.add_argument('--k', type=int, help='kernel width in model layers')
  parser.add_argument('--crop', type=int, default=0, help='amount to crop output image to remove border effects')

  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')

  parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')
  parser.add_argument('--training_file', type=str, default="/home/karima/cnn-data/subset7_100k_train_files.txt", help='filename of file with list of training data image files')
  parser.add_argument('--training_mosaic_file', type=str, default="/home/karima/cnn-data/xtrans-flat-subset7-100k-train.txt", help="filename of file with list of training data mosaic files")
  parser.add_argument('--validation_mosaic_file', type=str, default="/home/karima/cnn-data/xtrans-flat-subset7-100k-val.txt", help="filename of file with list of val data mosaic files")
  parser.add_argument('--test_mosaic_file', type=str, default="/home/karima/cnn-data/xtrans-flat-subset7-100k-test.txt", help="filename of file with list of test data mosaic files")

  parser.add_argument('--validation_file', type=str, default="/home/karima/cnn-data/val_files.txt", help='filename of file with list of validation data image files')
  parser.add_argument('--test_file', type=str, default="/home/karima/cnn-data/test_files.txt")

  parser.add_argument('--results_file', type=str, default='training_results', help='where to store training results')
  parser.add_argument('--validation_freq', type=int, default=None, help='validation frequency')

  parser.add_argument('--full_model', action="store_true")
  parser.add_argument('--testing', action='store_true')
  parser.add_argument('--train', action='store_true')

  parser.add_argument('--pretrained', action='store_true')
  parser.add_argument('--weights', type=str, help="pretrained weights")
  parser.add_argument('--green_pretrained', action='store_true')
  parser.add_argument('--green_weights', type=str, help="pretrained weights")

  args = parser.parse_args()


  args.train_input = "/home/karima/cnn-data/lowres-subset7-train.txt"
  args.train_target = "/home/karima/cnn-data/fullres-subset7-train.txt"

  args.val_input = "/home/karima/cnn-data/lowres-subset7-val.txt"
  args.val_target = "/home/karima/cnn-data/fullres-subset7-val.txt"

  args.test_input = "/home/karima/cnn-data/lowres-subset7-test.txt"
  args.test_target = "/home/karima/cnn-data/fullres-subset7-test.txt"


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
  args.width = [int(w.strip()) for w in args.width.split(",")]
  if len(args.width) == 1:
    args.width = args.width[0]

  if args.green_demosaicnet:
    logger.info("TRAINING DEMOSAICNET GREEN")
    model = superres_model_lib.GreenDemosaicknet(args.depth, args.width)

  if args.full_model and not (args.chromagradienthalide or args.fullrgbdemosaicnet or args.fullrgbgradienthalide):
    print(f'--- setting the no_grad parameter in green model {args.no_grad} ---')

    set_no_grad(green_model, args.no_grad)
    set_no_grad(model, args.no_grad)

    print('--- inserting green model ---')

    def insert_green(model):
      nodes = model.preorder()
      for n in nodes:
        if type(n) is demosaic_ast.Input:
          if n.name == "Input(GreenExtractor)":
            n.node = green_model
          elif hasattr(n, "node"): # other input ops my run submodels that also require the green model
            insert_green(n.node)
            
    insert_green(model)

  if args.green_pretrained:
    def set_green_weights(green_weights, model_ast):
      nodes = model_ast.preorder()
      for n in nodes:
        if type(n) is demosaic_ast.Input:
          if hasattr(n, 'node'):
            if hasattr(n, 'green_model_id'):
              n.weight_file = args.green_weights
            else:
              set_green_weights(green_weights, n.node)

    set_green_weights(args.green_weights, model)

  print(model.dump())

  ev = cost.ModelEvaluator(None)
  model_cost = ev.compute_cost(model, xtrans=True)
  print(f"model compute cost: {model_cost}")
  
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

  print('--- calling save ----')
  model_manager.save_model(torch_models, model, model_dir)

  for m in torch_models:
    m._initialize_parameters()
    
  for name, param in torch_models[0].named_parameters():
    print(f"{name} {param.size()}")

  if args.pretrained:
    weight_files = [f.strip for f in args.weights.split(',')]
    for i, m in enumerate(torch_models):
      state_dict = torch.load(weight_files[i])
      m.load_state_dict(state_dict)


  model_manager.save_model(torch_models, model, model_dir)

  validation_losses, training_losses = run(args, torch_models, 'seed', model_dir) 

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

