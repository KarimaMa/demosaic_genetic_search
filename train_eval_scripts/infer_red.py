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
from dataset import RedDataset, FastDataLoader

def run_infer(args, gr_model, gb_model, b_model): 
  gr_model = gr_model.cuda()
  gb_model = gb_model.cuda()
  b_model = b_model.cuda()

  gr_model.to_gpu(args.gpu)
  gb_model.to_gpu(args.gpu)
  b_model.to_gpu(args.gpu)

  criterion = nn.MSELoss()

  validation_data = RedDataset(data_file=args.validation_file, 
                          green_input=args.use_green_input, 
                          green_pred_input=args.use_green_pred,
                          green_file=args.green_validation_file,
                          all_red=True)
  
  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))

  validation_queue = FastDataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
      pin_memory=True, num_workers=8)

  valid_losses = infer(args, validation_queue, gr_model, gb_model, b_model, criterion)
  print(f'validation loss', valid_losses)

  return valid_losses
 

def infer(args, valid_queue, gr_model, gb_model, b_model, criterion):
  loss_tracker = util.AvgrageMeter()
  gr_model.eval()
  gb_model.eval()
  b_model.eval()

  bayer_mask = torch.from_numpy(np.zeros((1,128,128))).cuda()
  bayer_mask[0,0::2,1::2] = 1

  for step, (input, target) in enumerate(valid_queue): 
    quad, bayer, img = input 
    bayer = Variable(bayer, volatile=True).float().cuda()
    target = Variable(target, volatile=True).float().cuda()
    
    n = bayer.size(0) 
    print(n)

    model_inputs = {"Input(BayerQuad)": quad, "Input(Bayer)": bayer}
    gr_pred = gr_model.run(model_inputs)
    gb_pred = gb_model.run(model_inputs)
    b_pred = b_model.run(model_inputs)

    pred = gr_pred + gb_pred + b_pred + bayer * bayer_mask

    # print("gr")
    # print(gr_pred[0,:,0:4,0:4])

    # print("gb")
    # print(gb_pred[0,:,0:4,0:4])

    # print("b")
    # print(b_pred[0,:,0:4,0:4])

    # print("pred")
    # print(pred[0,:,0:4,0:4])

    # print("target")
    # print(target[0,:,0:4,0:4])
    # exit()

    loss = criterion(pred, target)
    loss_tracker.update(loss.item(), n)

  return loss_tracker.avg 


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--seed', type=int, default=3)

  parser.add_argument('--gr_model_weight', type=str)
  parser.add_argument('--gb_model_weight', type=str)
  parser.add_argument('--b_model_weight', type=str)

  parser.add_argument('--validation_file', type=str, default="/home/karima/cnn-data/val_files.txt", help='filename of file with list of validation data image files')

  parser.add_argument('--use_green_input', action="store_true")
  parser.add_argument('--full_model', action="store_true", default=True)
  parser.add_argument('--use_green_pred', action="store_true", help="whether to use precomputed green predictions")
  parser.add_argument('--green_training_file', type=str, help="filename of file with list of precomputed green for training data")
  parser.add_argument('--green_validation_file', type=str, help="filename of file with list of precomputed green for validation data")

  args = parser.parse_args()

  gr_model = model_lib.red_gr_simple_model()
  gb_model = model_lib.red_gb_simple_model()
  b_model = model_lib.red_b_simple_model()

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

  torch_gr_model = gr_model.ast_to_model().cuda() 
  torch_gr_model.load_state_dict(torch.load(args.gr_model_weight))

  torch_gb_model = gb_model.ast_to_model().cuda() 
  torch_gb_model.load_state_dict(torch.load(args.gb_model_weight))

  torch_b_model = b_model.ast_to_model().cuda() 
  torch_b_model.load_state_dict(torch.load(args.b_model_weight))

  validation_losses = run_infer(args, torch_gr_model, torch_gb_model, torch_b_model) 

