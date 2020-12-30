import torch
import torch.nn as nn
import torch.utils
import torchvision
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
import model_lib
from torch_model import ast_to_model, collect_operators
from dataset import GreenDataset, Dataset, FastDataLoader
from dataset import GreenQuadDataset, FullPredictionQuadDataset, FastDataLoader
from tree import print_parents
import demosaic_ast


def run(args, model, model_DAG):
  model.to_gpu(args.gpu)
 
  if not args.full_model:
    validation_data = GreenQuadDataset(data_file=args.validation_file)
  else:
    validation_data = FullPredictionQuadDataset(data_file=args.validation_file)

  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))

  validation_queue = FastDataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
      pin_memory=True, num_workers=8)
 
  visualize_outputs(args, validation_queue, model, model_DAG)


def visualize_outputs(args, valid_queue, model, model_DAG):
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      if args.full_model:
        bayer, redblue_bayer, green_grgb = input 
        redblue_bayer = Variable(redblue_bayer, requires_grad=False).float().cuda()
        green_grgb = Variable(green_grgb, requires_grad=False).float().cuda()
      else:
        bayer = input

      bayer = Variable(bayer, requires_grad=False).float().cuda()
      target = Variable(target, requires_grad=False).float().cuda()
      
      if args.crop > 0:
        target = target[..., args.crop:-args.crop, args.crop:-args.crop]

      n = bayer.size(0)

      model.reset()
      if args.full_model:
        model_inputs = {"Input(Bayer)": bayer, 
                        "Input(Green@GrGb)": green_grgb, 
                        "Input(RedBlueBayer)": redblue_bayer}
        pred = model.run(model_inputs)
      else:
        model_inputs = {"Input(Bayer)": bayer}
        pred = model.run(model_inputs)

      clamped = torch.clamp(pred, min=0, max=1)

      if args.crop > 0:
        clamped = clamped[..., args.crop:-args.crop, args.crop:-args.crop]
        
      model_operators = collect_operators(model)
      nodes = model_DAG.preorder()
      for nid, n in enumerate(nodes):
        if hasattr(model_operators[nid], "output") and not model_operators[nid].output is None:
          print(f"node {n.name} {type(model_operators[nid])} {model_operators[nid].output.size()}")
          tshape = list(model_operators[nid].output.shape)
          intermediate_output = model_operators[nid].output[0].reshape(-1, 1, tshape[2], tshape[3])
          torchvision.utils.save_image(intermediate_output, f"{args.outputdir}/{n.name}_{nid}.png")
        else:
          print(f"node {n.name} {type(model_operators[nid])}")

      torchvision.utils.save_image(clamped[0,...], f"{args.outputdir}/pred.png")
      torchvision.utils.save_image(target[0,...], f"{args.outputdir}/target.png")
      exit()

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

  parser.add_argument('--model_ast', type=str, help="model ast to use")
  parser.add_argument('--green_model_ast', type=str, help="model ast to use")

  parser.add_argument('--crop', type=int, default=0, help='amount to crop output image to remove border effects')

  parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--validation_file', type=str, default="/home/karima/cnn-data/subset7_100k_val_files.txt", help='filename of file with list of validation data image files')

  parser.add_argument('--full_model', action="store_true")

  parser.add_argument('--testing', action='store_true')

  parser.add_argument('--weights', type=str, help="pretrained weights")
  parser.add_argument('--green_weights', type=str, help="pretrained weights")

  parser.add_argument("--outputdir", type=str)

  args = parser.parse_args()

  os.mkdir(args.outputdir)
  model = demosaic_ast.load_ast(args.model_ast)

  ev = cost.ModelEvaluator(None)
  model_cost = ev.compute_cost(model)
  print(f"model compute cost: {model_cost}")

  if args.full_model:  
    green_model = demosaic_ast.load_ast(args.green_model_ast)

    nodes = model.preorder()
    for n in nodes:
      if type(n) is demosaic_ast.Input:
        if n.name == "Input(GreenExtractor)":
          n.weight_file = args.green_weights
          n.node = green_model

  if not torch.cuda.is_available():
    sys.exit(1)

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = False
  cudnn.enabled=True
  cudnn.deterministic=True

  torch_model = model.ast_to_model().cuda() 
 
  for name, param in torch_model.named_parameters():
    print(f"{name} {param.size()}")

  state_dict = torch.load(args.weights)
  torch_model.load_state_dict(state_dict)

  run(args, torch_model, model) 

