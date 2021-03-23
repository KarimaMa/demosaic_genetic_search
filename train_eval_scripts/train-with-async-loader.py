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
sys.path.append(sys.path[0].split("/")[0])

from dataset import FullPredictionProcessedDataset, GreenProcessedQuadDataset, FastDataLoader, FullPredictionQuadDataset, GreenQuadDataset
import util
from demosaic_ast import load_ast, get_green_model_id, Input
import cost
from async_loader import AsynchronousLoader


def run(args, model, model_id):
  if args.gpu >= 0: 
    model = model.cuda()
    model.to_gpu(args.gpu)
    
  if not args.full_model:
    if args.preprocessed:
      data = GreenProcessedQuadDataset(data_file=args.images, return_index=False)
    else:
      data = GreenQuadDataset(data_file=args.images, return_index=False)
  else:
    if args.preprocessed:
      data = FullPredictionProcessedDataset(data_file=args.images, RAM=args.ram, return_index=False)
    else:
      data = FullPredictionQuadDataset(data_file=args.images, return_index=False)

  train(args, data, model, model_id)


def train(args, data, model, model_id):
  device = torch.device(f'cuda:{args.gpu}')
  print(f"Using device {device}")

  num_train = len(data)

  if args.asyncload:
    loader = AsynchronousLoader(data, device, batch_size=args.batchsize, \
                                    pin_memory=True, workers=8, shuffle=True)
  else:
    loader = FastDataLoader(
        data, batch_size=args.batchsize, shuffle=True,
        pin_memory=True, num_workers=8)

  criterion = nn.MSELoss()
  model.train()

  optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

  backprop_time = 0
  forward_time = 0

  iter_ = 0

  with profiler.profile(use_cuda=True) as prof:
    with profiler.record_function("model_inference"):
      for epoch in range(args.epochs):
        for step, (input, target) in enumerate(loader):

          if args.full_model:
            bayer, redblue_bayer, green_grgb = input
            if not args.asyncload:
              bayer = bayer.cuda()
              redblue_bayer = redblue_bayer.cuda()
              green_grgb = green_grgb.cuda()
              target = target.cuda()
          else:
            bayer = input
            if not args.asyncload:
              bayer = bayer.cuda()
              target = target.cuda()

          if iter_ == 1:
            start = time.perf_counter()

          n = bayer.size(0)
          optimizer.zero_grad()
          model.reset()

          forward_start = time.perf_counter()
          if args.full_model:
            model_inputs = {"Input(Bayer)": bayer, 
                            "Input(Green@GrGb)": green_grgb, 
                            "Input(RedBlueBayer)": redblue_bayer}
            pred = model.run(model_inputs)
          else:
            model_inputs = {"Input(Bayer)": bayer}
            pred = model.run(model_inputs)
          
          forward_end = time.perf_counter()
          if iter_ > 0:
            forward_time += (forward_end - forward_start)

          backprop_start = time.perf_counter()
          target = target[..., args.crop:-args.crop, args.crop:-args.crop]
          pred = pred[..., args.crop:-args.crop, args.crop:-args.crop]
          loss = criterion(target, pred)
        
          loss.backward()
          backprop_end = time.perf_counter()
          if iter_ > 0:
            backprop_time += (backprop_end - backprop_start)

          # if step % 500 == 0:
          #   print(f"{loss.item()}")
          #   # compute batch average psnr
          #   clamped = torch.clamp(pred, min=0, max=1).detach()
          #   per_image_mse = (clamped-target).square().mean(-1).mean(-1).mean(-1)
          #   per_image_psnr = -10.0*torch.log10(per_image_mse)
          #   batch_avg_psnr = per_image_psnr.sum(0) / n
          #   print(f"{loss.item()} psnr: {batch_avg_psnr}")

          optimizer.step()

          iter_ += 1

  end = time.perf_counter()
  print(f"step {step}")
  print(f"time per batch: {(end - start)/(iter_-1)}")
  print(f"time per backprop: {backprop_time/(iter_-1)}")
  print(f"time per forward: {forward_time/(iter_-1)}")

  prof.export_chrome_trace(f"model-{model_id}-aync-{args.asyncload}-epochs-{args.epochs}-train-trace.json")
  #print(prof.key_averages().table(row_limit=25))


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
  parser.add_argument('--batchsize', type=int, default=64)
  parser.add_argument('--learning_rate', type=float, default=0.003)
  parser.add_argument('--epochs', type=int)
  parser.add_argument('--gpu', type=int, default=-1)
  
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

  if args.gpu >= 0:  
    torch_model = model_ast.ast_to_model().cuda() 
  else:
    torch_model = model_ast.ast_to_model()

  torch_model._initialize_parameters()
  
  run(args, torch_model, model_id)
