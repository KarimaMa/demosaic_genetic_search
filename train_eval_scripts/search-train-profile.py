"""
Compares training implementation as it was carried out in 
the search code with the optimized new version
"""


import argparse
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.utils
import torch.autograd.profiler as profiler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os
import numpy as np
import sys
import re
import csv
import logging
sys.path.append(sys.path[0].split("/")[0])

from dataset import FullPredictionProcessedDataset, GreenProcessedQuadDataset, FastDataLoader, FullPredictionQuadDataset, GreenQuadDataset
import util
from demosaic_ast import load_ast, get_green_model_id, Input
import cost
from async_loader import AsynchronousLoader


def run(args, models, model_id):
  if args.gpu >= 0: 
    models = [m.cuda() for m in models]
    for m in models:
      m.to_gpu(args.gpu)
    
  if not args.full_model:
    if args.preprocessed:
      data = GreenProcessedQuadDataset(data_file=args.images, return_index=False)
    else:
      data = GreenQuadDataset(data_file=args.images, return_index=False)
  else:
    if args.preprocessed:
      data = FullPredictionProcessedDataset(data_file=args.images, RAM=args.ram, return_index=False)
    else:
      data = FullPredictionQuadDataset(data_file=args.images, RAM=args.ram, return_index=False)

  if args.old_code:
    old_train(args, data, models, model_id)
  else:
    new_train(args, data, models, model_id)



def create_loggers(num_models):
  log_format = '%(asctime)s %(levelname)s %(message)s'
  log_files = [f'v{i}_log' for i in range(num_models)]
  for log_file in log_files:
    if os.path.exists(log_file):
      os.remove(log_file)
  loggers = [util.create_logger(f'v{i}_logger', logging.INFO, log_format, log_files[i]) for i in range(num_models)]
  return loggers


def old_train(args, data, models, model_id):
  device = torch.device(f'cuda:{args.gpu}')
  print(f"Using device {device}")
  loss_trackers = [util.AvgrageMeter() for m in models]
  psnr_trackers = [util.AvgrageMeter() for m in models]

  train_loggers = create_loggers(args.model_inits)

  for m in models:
    m.train()

  optimizers = [torch.optim.Adam(m.parameters(), args.learning_rate) for m in models]
  criterion = nn.MSELoss()

  loader = FastDataLoader(
        data, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=4)

  iter_ = 0

  for epoch in range(args.epochs):
    for step, (input, target) in enumerate(loader):
      if step % 200 == 0:
        print(step)
        
      if args.full_model:
        bayer, redblue_bayer, green_grgb = input 
        redblue_bayer = Variable(redblue_bayer, requires_grad=False).to(device=f"cuda:{args.gpu}", dtype=torch.float)
        green_grgb = Variable(green_grgb, requires_grad=False).to(device=f"cuda:{args.gpu}", dtype=torch.float)
      else:
        bayer = input

      bayer = Variable(bayer, requires_grad=False).to(device=f"cuda:{args.gpu}", dtype=torch.float)
      target = Variable(target, requires_grad=False).to(device=f"cuda:{args.gpu}", dtype=torch.float)
      
      if iter_ == 1:
        start = time.perf_counter()

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
        # compute running psnr
        per_image_mse = (pred-target).square().mean(-1).mean(-1).mean(-1)
        per_image_psnr = -10.0*torch.log10(per_image_mse)
        batch_avg_psnr = per_image_psnr.sum(0) / n
        psnr_trackers[i].update(batch_avg_psnr.item(), n)

        if step % args.report_freq == 0 or step == len(loader)-1:
          train_loggers[i].info('train %03d %e %2.3f', epoch*len(loader)+step, loss.item(), batch_avg_psnr)

      iter_ += 1

  end = time.perf_counter()
  print(f"step {step}")
  print(f"time per batch: {(end - start)/(iter_-1)}")
 

def new_train(args, data, models, model_id):
  device = torch.device(f'cuda:{args.gpu}')
  print(f"Using device {device}")
  loss_trackers = [util.AvgrageMeter() for m in models]
  train_loggers = create_loggers(args.model_inits)

  num_train = len(data)

  loader = FastDataLoader(
        data, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=8)

  if args.asyncload:
    loader = AsynchronousLoader(loader, device)
 
  criterion = nn.MSELoss()
  for m in models:
    m.train()

  optimizers = [torch.optim.Adam(m.parameters(), args.learning_rate) for m in models]

  iter_ = 0

  for epoch in range(args.epochs):
    for step, (input, target) in enumerate(loader):
      if step % 200 == 0:
        print(step)
      if args.full_model:
        bayer, redblue_bayer, green_grgb = input 
      else:
        bayer = input
       
      if iter_ == 1:
        start = time.perf_counter()

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

        # if step % args.report_freq == 0 or step == len(loader)-1:
        #   train_loggers[i].info(f"train step {epoch*len(loader)+step} loss {loss.item():.5f}")

      iter_ += 1
    # if step % args.save_freq == 0 or step == len(train_queue)-1:
    #   model_pytorch_files = [util.get_model_pytorch_file(model_dir, model_version, epoch) for model_version in range(len(models))]
    #   for i in range(len(models)):
    #     torch.save(models[i].state_dict(), model_pytorch_files[i])
   
  end = time.perf_counter()
  print(f"step {step}")
  print(f"time per batch: {(end - start)/(iter_-1)}")
 

  # prof.export_chrome_trace(f"model-{model_id}-aync-{args.asyncload}-epochs-{args.epochs}-train-trace.json")
  # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))


"""
Inserts green ast into chroma ast and sets 
no_grad for all sub models inside the chroma ast
"""
def insert_green_model(args, chroma_ast, green_ast, no_grad):
  nodes = chroma_ast.preorder()
  for n in nodes:
    if type(n) is Input:
      if n.name == "Input(GreenExtractor)":
        n.node = green_ast
        n.no_grad = no_grad
      elif hasattr(n, "node"): # other input ops my run submodels that also require the green model
        insert_green_model(args, n.node, green_ast, no_grad)
        n.no_grad = no_grad


def set_green_weights(green_weights, model_ast):
  nodes = model_ast.preorder()
  for n in nodes:
    if type(n) is Input:
      if hasattr(n, 'node'):
        if hasattr(n, 'green_model_id'):
          n.weight_file = green_weights
        else:
          set_green_weights(green_weights, n.node)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--modeldir', type=str, help='where model asts, training logs, and weights are stored')
  parser.add_argument('--model_id', type=int, help='model id from modeldir to run')
  parser.add_argument('--crop', type=int, default=16)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--learning_rate', type=float, default=0.003)
  parser.add_argument('--epochs', type=int)
  parser.add_argument('--gpu', type=int, default=-1)
  parser.add_argument('--model_inits', type=int, default=3)
  parser.add_argument('--report_freq', type=int, default=200)
  
  parser.add_argument('--old_code', action="store_true")

  parser.add_argument('--testing', action='store_true')
  # training parameters
  parser.add_argument('--images', type=str, help='filename of file with list of data image files')
  
  # training full chroma + green parameters
  parser.add_argument('--full_model', action="store_true")
  parser.add_argument('--modular', action="store_true")

  parser.add_argument('--green_model_asts', type=str, help="file with list of filenames of green model asts used as inputs to chroma models")
  parser.add_argument('--green_weights', type=str, help="pretrained weights")

  parser.add_argument('--asyncload', action='store_true')
  parser.add_argument('--ram', action='store_true')
  parser.add_argument('--preprocessed', action='store_true')

  args = parser.parse_args()

  if args.gpu >= 0 and not torch.cuda.is_available():
    sys.exit(1)
  
  torch.cuda.set_device(args.gpu)

  if args.full_model and args.modular:
    args.green_model_asts = [l.strip() for l in open(args.green_model_asts)]
    args.green_model_weights = [l.strip() for l in open(args.green_weights)]

  model_id = args.model_id
  print(f"running model {model_id}")
  model_info_dir = os.path.join(args.modeldir, f"{model_id}")
  model_ast_file = os.path.join(model_info_dir, "model_ast")

  model_ast = load_ast(model_ast_file)

  if args.full_model and args.modular:
    green_model_id = get_green_model_id(model_ast)
    green_model_ast_file = args.green_model_asts[green_model_id]
    green_model_weights = args.green_model_weights[green_model_id]

    green_model = load_ast(green_model_ast_file)
    
    no_grad = not (args.green_weights is None) 
    print(f"updating green weights? {no_grad}")
    insert_green_model(args, model_ast, green_model, no_grad)

    if no_grad:
      set_green_weights(green_model_weights, model_ast)

  torch_models = [model_ast.ast_to_model() for i in range(args.model_inits)]

  for m in torch_models:
    m._initialize_parameters()
  
  run(args, torch_models, model_id)
