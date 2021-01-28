import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os
import numpy as np
import sys
import re
import csv

sys.path.append(sys.path[0].split("/")[0])

from dataset import GradHalideDataset, FastDataLoader
import util
import gradienthalide_models


def run(args, model):
  model = model.cuda()
  model.to_gpu(args.gpu)
    
  test_data = GradHalideDataset(data_file=args.test_file)

  infer(args, test_data, model)


def infer(args, test_data, model):
  num_test = len(test_data)
  test_indices = list(range(num_test))

  test_queue = FastDataLoader(
    test_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SequentialSampler(test_indices),
    pin_memory=True, num_workers=4)
  
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(test_queue):
      flatb, threechanb = input
      flatb = flatb.to(device=f"cuda:{args.gpu}")
      threechanb = threechanb.to(device=f"cuda:{args.gpu}")
      input = (flatb, threechanb) 
      target = target.to(device=f"cuda:{args.gpu}")
   
      n = target.size(0)
   
      pred = model(input)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--k', type=int)
  parser.add_argument('--filters', type=int)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--gpu', type=int, default=0)
  
  parser.add_argument('--test_file', type=str, help='filename of file with list of data image files to evaluate')

  args = parser.parse_args()

  if not torch.cuda.is_available():
    sys.exit(1)

  torch_model = gradienthalide_models.GradientHalide(args.k, args.filters)
  run(args, torch_model)




