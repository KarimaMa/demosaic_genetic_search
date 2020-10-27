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
from demosaic_ast import load_ast
from dataset import Dataset, ids_from_file, FastDataLoader
import numpy as np
from torch.autograd import Variable



def precompute_training_green(args, gpu_id, model):
  model.eval()
  full_data_filenames = ids_from_file(args.training_file)
  used_filenames = full_data_filenames[0:int(args.train_portion)]

  train_data = Dataset(data_filenames=used_filenames, RAM=False, return_index=True,\
                      green_input=False, green_output=True)
  num_train = len(train_data)
  train_indices = list(range(num_train))
  
  train_queue = FastDataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=True, num_workers=8)

  criterion = torch.nn.MSELoss()

  with torch.no_grad():
    for step, (index, input, target) in enumerate(train_queue):
      bayer = input
      bayer_input = Variable(bayer, requires_grad=False).to(device=f"cuda:{gpu_id}")
      target = Variable(target, requires_grad=False).to(device=f"cuda:{gpu_id}")

      model_inputs = {"Input(Bayer)": bayer_input}
      pred = model.run(model_inputs)
      loss = criterion(pred, target)
      if step % 100 == 0:
        print(f"loss {loss}")

      for i,idx in enumerate(index):
        input_filename = used_filenames[idx]
        if args.satori:
          subpath = ('/').join(input_filename.split('/')[-2:])
        else:
          subpath = ('/').join(input_filename.split('/')[-4:])
    
        subpath = subpath.strip('png')
        subpath += "pytensor"
        subdir = os.path.join(args.green_data_dir, ("/").join(subpath.split("/")[0:-1]))
        os.makedirs(subdir, exist_ok=True)
        pred_filename = os.path.join(subdir, subpath.split("/")[-1])
        torch.save(pred[i].clone(), pred_filename)


def precompute_validation_green(args, gpu_id, model):
  model.eval()
  validation_data = Dataset(data_file=args.validation_file, RAM=False, return_index=True,\
                            green_input=False, green_output=True)
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
      bayer = input
      bayer_input = Variable(bayer, requires_grad=False).to(device=f"cuda:{gpu_id}")
      target = Variable(target, requires_grad=False).to(device=f"cuda:{gpu_id}")

      model_inputs = {"Input(Bayer)": bayer_input}
      pred = model.run(model_inputs)
      loss = criterion(pred, target)

      if step % 100 == 0:
        print(f"loss {loss}")

      for i, idx in enumerate(index):
        input_filename = used_filenames[idx]
        if args.satori:
          subpath = ('/').join(input_filename.split('/')[-2:])
        else:
          subpath = ('/').join(input_filename.split('/')[-4:])

        subpath = subpath.strip('png')
        subpath += "pytensor"
        
        subdir = os.path.join(args.green_data_dir, ("/").join(subpath.split("/")[0:-1]))
        os.makedirs(subdir, exist_ok=True)
        pred_filename = os.path.join(subdir, subpath.split("/")[-1])
        torch.save(pred[i].clone(), pred_filename)


def run(args, gpu_id):
  """
  model_id = 0 
  model_dir = args.model_path
  util.create_dir(model_dir)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  training_logger = util.create_logger(f'model_green_train_logger', logging.INFO, log_format, \
                                        os.path.join(model_dir, f'model_green_training_log'))
  """

  model_ast = load_ast(args.green_model_ast_file)
  #pytorch_models = [model_ast.ast_to_model() for i in range(args.model_initializations)]
  pytorch_model = model_ast.ast_to_model()
  pytorch_model.load_state_dict(torch.load(args.green_model_weight_file))
  pytorch_model = pytorch_model.to(device=f"cuda:{gpu_id}") 
  pytorch_model.to_gpu(gpu_id)
    
  #model_valid_psnrs, model_train_psnrs = train_model(args, gpu_id, model_id, pytorch_models, model_dir, training_logger)

  #best_model = pytorch_models[np.argmax(model_valid_psnrs)]

  precompute_training_green(args, gpu_id, pytorch_model)
  precompute_validation_green(args, gpu_id, pytorch_model)
 


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--model_path', type=str, default='models', help='where to save training results')
  parser.add_argument('--green_model_ast_file', type=str, help='ast file for green model')
  parser.add_argument('--green_model_weight_file', type=str, help='torch file with greem model weights')
  #parser.add_argument('--save', type=str, help='experiment name')
  parser.add_argument('--green_data_dir', type=str, help='where to save green precomputed predictions')
  parser.add_argument('--satori', action='store_true')
  parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')
  parser.add_argument('--train_portion', type=int, default=1e5, help='portion of training data to use')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')

  # training parameters
  """
  parser.add_argument('--seed', type=int, default=2)
  parser.add_argument('--num_gpus', type=int, default=4, help='number of available GPUs') # change this to use all available GPUs
  parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
  parser.add_argument('--lr_search_steps', type=int, default=1, help='how many line search iters for finding learning rate')
  parser.add_argument('--variance_min', type=float, default=0.003, help='minimum validation psnr variance')
  parser.add_argument('--variance_max', type=float, default=0.02, help='maximum validation psnr variance')
  parser.add_argument('--weight_decay', type=float, default=1e-16, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=200, help='training report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='trained weights save frequency')
  parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--validation_freq', type=int, default=50, help='validation frequency for assessing validation PSNR variance')
  """

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

  run(args, 0)

