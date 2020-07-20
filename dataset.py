import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data
import random
from imageio import imread
from config import IMG_H, IMG_W


"""
from mgharbi demosaicnet code
"""
def bayer(im, return_mask=False):
  """Bayer mosaic.
  The patterned assumed is::
  G r
  b G

  Args:
  im (np.array): image to mosaic. Dimensions are [c, h, w]
  return_mask (bool): if true return the binary mosaic mask, instead of the mosaic image.

  Returns:
  np.array: mosaicked image (if return_mask==False), or binary mask if (return_mask==True)
  """

  mask = np.ones_like(im)

  if return_mask:
    return mask

  # red
  mask[0, ::2, 0::2] = 0
  mask[0, 1::2, :] = 0

  # green
  mask[1, ::2, 1::2] = 0
  mask[1, 1::2, ::2] = 0

  # blue
  mask[2, 0::2, :] = 0
  mask[2, 1::2, 1::2] = 0

  return im*mask


def ids_from_file(filename):
  ids = [l.strip() for l in open(filename, "r")]
  return ids


class Dataset(data.Dataset):
  def __init__(self, train_filenames):
    self.list_IDs = ids_from_file(train_filenames) # patch filenames

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):

    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])

    mosaic = bayer(img)

    return (mosaic, img) 


class GreenDataset(data.Dataset):
  def __init__(self, train_filenames):
    self.list_IDs = ids_from_file(train_filenames) # patch filenames

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):

    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])
    green = np.expand_dims(img[1,...], axis=0)
    mosaic = bayer(img)
    mosaic = np.sum(mosaic, axis=0, keepdims=True)
    return (mosaic, green) 
