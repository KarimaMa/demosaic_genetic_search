"""
Given green model to use, trains it on dataset and then
uses trained model to generate predicted green dataset to use 
for training chroma models
"""
import math 
import logging
import sys
import os
import argparse
import glob
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
sys.path.append(sys.path[0].split("/")[0])
import util
from train import train_model
from dataset import GreenQuadDataset, ids_from_file, FastDataLoader
import numpy as np
from torch.autograd import Variable
from demosaicnet_models import *
from experimental_green_models import *


def precompute_training_green(args, model):
  model.eval()
  full_data_filenames = ids_from_file(args.training_file)
  used_filenames = full_data_filenames[0:int(args.train_portion)]

  train_data = GreenQuadDataset(data_file=args.training_file, return_index=True) 

  num_train = len(train_data)
  train_indices = list(range(num_train))
  
  train_queue = FastDataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=True, num_workers=8)

  criterion = torch.nn.MSELoss()

  with torch.no_grad():
    for step, (index, input, target) in enumerate(train_queue):
      bayer_input = Variable(input, requires_grad=False).float().to(device=f"cuda:{args.gpu}")
      target = Variable(target, requires_grad=False).float().to(device=f"cuda:{args.gpu}")

      pred = model(bayer_input)
      print("pred")
      print(pred[0,:,0:4,0:4])
      exit()
      
      loss = criterion(pred, target)
      if step % 100 == 0:
        print(f"loss {loss}")

      for i,idx in enumerate(index):
        input_filename = used_filenames[idx]
        subpath = ('/').join(input_filename.split('/')[-4:])   
        subpath = subpath.strip('png')
        subpath += "npy"
        subdir = os.path.join(args.green_data_dir, ("/").join(subpath.split("/")[0:-1]))
        os.makedirs(subdir, exist_ok=True)
        pred_filename = os.path.join(subdir, subpath.split("/")[-1])
        np.save(pred_filename, pred[i].clone().cpu().numpy())


def precompute_validation_green(args, model):
  model.eval()
  validation_data = GreenQuadDataset(data_file=args.validation_file, return_index=True)
  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))
  used_filenames = validation_data.list_IDs 

  validation_queue = FastDataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
      pin_memory=True, num_workers=8)

  criterion = torch.nn.MSELoss()

  with torch.no_grad():
    for step, (index, input, target) in enumerate(validation_queue):
      bayer_input = Variable(input, requires_grad=False).float().to(device=f"cuda:{args.gpu}")
      target = Variable(target, requires_grad=False).float().to(device=f"cuda:{args.gpu}")

      pred = model(bayer_input)
      loss = criterion(pred, target)

      if step % 100 == 0:
        print(f"loss {loss}")

      for i, idx in enumerate(index):
        input_filename = used_filenames[idx]
        subpath = ('/').join(input_filename.split('/')[-4:])
        subpath = subpath.strip('png')
        subpath += "npy"
        
        subdir = os.path.join(args.green_data_dir, ("/").join(subpath.split("/")[0:-1]))
        os.makedirs(subdir, exist_ok=True)
        pred_filename = os.path.join(subdir, subpath.split("/")[-1])
        np.save(pred_filename, pred[i].clone().cpu().numpy())


def run(args):
  if args.multiresquad:
    pytorch_model = MultiresQuadGreenModel(depth=args.depth, width=args.width, scale_factor=args.scale_factor, bias=args.bias)
  else:
    pytorch_model = GreenDemosaicknet(depth=args.depth, width=args.width)

  pytorch_model.load_state_dict(torch.load(args.green_model_weight_file))
  pytorch_model = pytorch_model.to(device=f"cuda:{args.gpu}") 
  if not args.multiresquad:
    pytorch_model.to_gpu(args.gpu)

  precompute_training_green(args, pytorch_model)
  precompute_validation_green(args, pytorch_model)
 


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Compute Green with Demosaicnet")
  parser.add_argument('--green_model_weight_file', type=str, help='torch file with greem model weights')
  parser.add_argument('--green_data_dir', type=str, help='where to save green precomputed predictions')
  parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')
  parser.add_argument('--train_portion', type=int, default=1e5, help='portion of training data to use')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--width', type=int)
  parser.add_argument('--depth', type=int)
  parser.add_argument('--bias', action='store_true')
  parser.add_argument('--scale_factor', type=int)
  parser.add_argument('--multiresquad', action='store_true')
  parser.add_argument('--gpu', type=int)


  args = parser.parse_args()
  args.use_green_input=False
  args.full_model=False

  if not torch.cuda.is_available():
    sys.exit(1)

  """
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  """

  cudnn.benchmark = False
  cudnn.enabled=True
  cudnn.deterministic=True
  #torch.cuda.manual_seed(args.seed)

  #util.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  #args.model_path = os.path.join(args.save, args.model_path)

  os.makedirs(args.green_data_dir, exist_ok=True)

  run(args)

