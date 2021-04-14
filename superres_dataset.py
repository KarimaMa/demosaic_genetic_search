import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data
from imageio import imread
from config import IMG_H, IMG_W
from util import ids_from_file
from superres_mosaic_gen import lowres_bayer



class GreenQuadDataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None, return_index=False):
    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames

    self.return_index = return_index

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])
    img = torch.Tensor(img).unsqueeze(0)
    lowres_mosaic = lowres_bayer(img)
    lowres_mosaic = torch.sum(lowres_mosaic, 0, keepdim=True)

    quad_size = list(lowres_mosaic.shape)
    quad_size[0] = 4
    quad_size[1] //= 2
    quad_size[2] //= 2

    bayer_quad = np.zeros(quad_size)
    bayer_quad[0,:,:] = lowres_mosaic[0,0::2,0::2]
    bayer_quad[1,:,:] = lowres_mosaic[0,0::2,1::2]
    bayer_quad[2,:,:] = lowres_mosaic[0,1::2,0::2]
    bayer_quad[3,:,:] = lowres_mosaic[0,1::2,1::2]

    bayer_quad = torch.Tensor(bayer_quad)
    input = bayer_quad
  
    target = np.expand_dims(img[0,1,...], axis=0)
    target = torch.Tensor(target)

    if self.return_index:
      return (index, input, target)
  
    return (input, target) 