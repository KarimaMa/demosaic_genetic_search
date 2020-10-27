import torch
import torch.nn as nn
import torch.utils
from torch.autograd import Variable
import argparse
import os
import random 
import numpy as np
import sys
sys.path.append(sys.path[0].split("/")[0])
from dataset import Dataset, ids_from_file, FastDataLoader
import util



def compare(args):
  val_filenames = ids_from_file(args.validation_file)
  validation_data = Dataset(data_file=args.validation_file, RAM=False, return_index=True,\
                            green_input=False, green_pred_input=False,\
                            green_output=True)
  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))

  validation_queue = FastDataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
      pin_memory=True, num_workers=8)

  print(f"FINISHED creating datasets")

  criterion = nn.MSELoss()
  loss_tracker = util.AvgrageMeter() 

  green_list_IDs = ids_from_file(args.green_data_file) # patch filenames

  for step, (index, input, target) in enumerate(validation_queue):
    target = Variable(target, requires_grad=False)

    n = input.size(0)

    green_batch = torch.empty(target.shape)
    for i, idx in enumerate(index):
      green_f = green_list_IDs[idx]
      green = np.load(green_f)
      green_batch[i,...] = torch.from_numpy(green)

    loss = criterion(green_batch, target)
    loss_tracker.update(loss.item(), n)

    if step % 50 == 0 or step == len(validation_queue)-1:
      print('validation %03d %e', len(validation_queue)+step, loss.item())

  print(f"average loss: {loss_tracker.avg} psnr: {util.compute_psnr(loss_tracker.avg)}")


parser = argparse.ArgumentParser()
parser.add_argument("--validation_file", type=str)
parser.add_argument("--green_data_file", type=str)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

compare(args)
