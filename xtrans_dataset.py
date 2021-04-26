import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data
from imageio import imread
from config import IMG_H, IMG_W
from util import ids_from_file
from mosaic_gen import bayer, xtrans, xtrans_3x3_invariant, xtrans_cell
from skimage import color



class XGreenDataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None, flat=False, return_index=False):
    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames

    self.mask = xtrans_cell(h=120, w=120)
    self.flat = flat
    self.return_index = return_index

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])
    img = img[:,4:-4,4:-4]

    mosaic3chan = xtrans(img, mask=self.mask)
    flat_mosaic = np.sum(mosaic3chan, axis=0, keepdims=True)
    packed_mosaic = xtrans_3x3_invariant(img)

    input = packed_mosaic, mosaic3chan, flat_mosaic
  
    target = np.expand_dims(img[1,...], axis=0)
    target = torch.Tensor(target)

    if self.return_index:
      return (index, input, target)
  
    return (input, target) 
 


class XRGBDataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None, precomputed=None, flat=False, return_index=False):
    if data_file:
      self.list_IDs = sort_ids(ids_from_file(data_file)) # patch filenames
    else:
      self.list_IDs = data_filenames

    if precomputed: # read precomputed mosaic
      self.mosaic_IDs = sort_ids(ids_from_file(precomputed))
      self.precomputed = True
    else:
      self.precomputed = False
      self.mask = xtrans_cell(h=120, w=120)

    self.flat = flat
    self.return_index = return_index

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])
    img = img[:,4:-4,4:-4]

    mosaic3chan = xtrans(img, mask=self.mask)
    flat_mosaic = np.sum(mosaic3chan, axis=0, keepdims=True)
    packed_mosaic = xtrans_3x3_invariant(img)

    input = packed_mosaic, mosaic3chan, flat_mosaic
  
    target = img
    target = torch.Tensor(target)

    if self.return_index:
      return (index, input, target)
  
    return (input, target) 
 

 