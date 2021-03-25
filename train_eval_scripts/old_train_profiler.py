import argparse
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

from old_dataset import FullPredictionQuadDataset, GreenQuadDataset, RGB8ChanDataset, ids_from_file, FastDataLoader
import util
from demosaic_ast import load_ast, get_green_model_id, Input
import cost
import time

def run(args, model, model_id):
  if args.gpu >= 0: 
    model = model.cuda()
    model.to_gpu(args.gpu)
    
  if not args.full_model:
    data = GreenQuadDataset(data_file=args.images, return_index=True)
  else:
    data = FullPredictionQuadDataset(data_file=args.images, return_index=True)

  infer(args, data, model, model_id)


def infer(args, data, model, model_id):
  num_test = len(data)
  indices = list(range(num_test))

  loader = FastDataLoader(
    data, batch_size=args.batchsize,
    sampler=torch.utils.data.sampler.RandomSampler(indices),
    pin_memory=False, num_workers=4)
  
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

  model.train()

  for epoch in range(args.epochs):
    for step, (input_img_names, input, target) in enumerate(loader):
      if step % 100 == 0:
        print(f"step {step}")
      if args.full_model:
        bayer, redblue_bayer, green_grgb = input 
        redblue_bayer = redblue_bayer.float()
        green_grgb = green_grgb.float()
        if args.gpu >= 0:
          redblue_bayer = redblue_bayer.cuda()
          green_grgb = green_grgb.cuda()
      else:
        bayer = input
        
      bayer = bayer.float()
      target = target.float()
      if args.gpu >= 0:
        bayer = bayer.cuda()
        target = target.cuda()
      
      if step == 0 and epoch == 1:
          start = time.perf_counter()

      target = target[..., args.crop:-args.crop, args.crop:-args.crop]

      n = bayer.size(0)
      optimizer.zero_grad()

      model.reset()
      if args.full_model:
        model_inputs = {"Input(Bayer)": bayer, 
                        "Input(Green@GrGb)": green_grgb, 
                        "Input(RedBlueBayer)": redblue_bayer}
        pred = model.run(model_inputs)
      else:
        model_inputs = {"Input(Bayer)": bayer}
        pred = model.run(model_inputs)

      pred = pred[..., args.crop:-args.crop, args.crop:-args.crop]
      loss = criterion(pred, target)
      loss.backward()
      optimizer.step()

  end = time.perf_counter()
  print(f"time per batch: {(end - start)/(len(loader) * (args.epochs-1))}")

"""
Inserts green ast into chroma ast and sets 
no_grad for all sub models inside the chroma ast
"""
def insert_green_model(args, chroma_ast, green_ast):
  nodes = chroma_ast.preorder()
  for n in nodes:
    if type(n) is Input:
      if n.name == "Input(GreenExtractor)":
        n.node = green_ast
        n.no_grad = True
      elif hasattr(n, "node"): # other input ops my run submodels that also require the green model
        insert_green_model(args, n.node, green_ast)
        n.no_grad = True


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--modeldir', type=str, help='where model asts, training logs, and weights are stored')
  parser.add_argument('--model_id', type=int, help='model id from modeldir to run')
  parser.add_argument('--crop', type=int, default=16)
  parser.add_argument('--batchsize', type=int, default=64)
  parser.add_argument('--gpu', type=int, default=-1)
  parser.add_argument('--epochs', type=int)
  parser.add_argument('--learning_rate', type=float, default=0.003)
  
  # training parameters
  parser.add_argument('--images', type=str, help='filename of file with list of data image files to evaluate')
  
  # training full chroma + green parameters
  parser.add_argument('--full_model', action="store_true")
  parser.add_argument('--modular', action="store_true")

  parser.add_argument('--green_model_asts', type=str, help="file with list of filenames of green model asts used as inputs to chroma models")

  args = parser.parse_args()

  if args.gpu >= 0 and not torch.cuda.is_available():
    sys.exit(1)
  
  torch.cuda.set_device(args.gpu)

  if args.full_model and args.modular:
    args.green_model_asts = [l.strip() for l in open(args.green_model_asts)]

  model_id = args.model_id
  print(f"running model {model_id}")
  model_info_dir = os.path.join(args.modeldir, f"{model_id}")
  model_ast_file = os.path.join(model_info_dir, "model_ast")

  model_ast = load_ast(model_ast_file)

  if args.full_model and args.modular:
    green_model_id = get_green_model_id(model_ast)
    green_model_ast_file = args.green_model_asts[green_model_id]
    green_model = load_ast(green_model_ast_file)
    
    insert_green_model(args, model_ast, green_model)

    print(model_ast.dump())

  if args.gpu >= 0:  
    torch_model = model_ast.ast_to_model().cuda() 
  else:
    torch_model = model_ast.ast_to_model()

  torch_model._initialize_parameters()
  
  run(args, torch_model, model_id)
