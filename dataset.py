import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data
import random
from imageio import imread
from config import IMG_H, IMG_W


class _RepeatSampler(object):
  """ Sampler that repeats forever.
  Args:
  sampler (Sampler)
  """
  def __init__(self, sampler):
    self.sampler = sampler

  def __iter__(self):
    while True:
      yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
    self.iterator = super().__iter__()

  def __len__(self):
    return len(self.batch_sampler.sampler)

  def __iter__(self):
    for i in range(len(self)):
      yield next(self.iterator)


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

  # load entire dataset into mp.Array 
  def __init__(self, data_file=None, data_filenames=None, return_index=False):
    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames
    self.return_index = return_index
    self.use_cropping = use_cropping
    self.RAM = RAM
    
    if self.RAM:
      self.inputs = []
      self.labels = []
      for index in range(len(self.list_IDs)):
        if index % 10000 == 0:
          print(f"loading {index}")
        image_f = self.list_IDs[index]
        img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
        img = np.transpose(img, [2, 0, 1])
        green = np.expand_dims(img[1,...], axis=0)
        mosaic = bayer(img)
        mosaic = np.sum(mosaic, axis=0, keepdims=True)

        if self.use_cropping: # take center crop of image
          mosaic = mosaic[...,32:94,32:94]
          green = green[...,32:94,32:94]

        self.inputs.append(mosaic)
        self.labels.append(green)

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    if self.RAM:
      if self.return_index:
        return (index, self.inputs[index], self.labels[index])
      else:
        return (self.inputs[index], self.labels[index])

    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])
    green = np.expand_dims(img[1,...], axis=0)
    mosaic = bayer(img)
    mosaic = np.sum(mosaic, axis=0, keepdims=True)
  
    if self.return_index:
      return (index, mosaic, green)
    if self.use_cropping: # take center crop of image
      mosaic = mosaic[...,32:94,32:94]
      green = green[...,32:94,32:94]
    return (mosaic, green) 


def load_data_to_RAM(data_file=None, data_filenames=None):
  if data_file:
    list_IDs = ids_from_file(data_file) # patch filenames
  else:
    list_IDs = data_filenames
  
  inputs = []
  labels = []
  for index in range(len(list_IDs)):
    if index % 10000 == 0:
      print(f"loading {index}")
    image_f = list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])
    green = np.expand_dims(img[1,...], axis=0)
    mosaic = bayer(img)
    mosaic = np.sum(mosaic, axis=0, keepdims=True)

    inputs.append(mosaic)
    labels.append(green)

  return np.array(inputs), np.array(labels)



class GreenSharedDataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None, inputs=None, labels=None):
    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames
    self.inputs = inputs
    self.labels = labels

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    return (self.inputs[index], self.labels[index])


class GreenDataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None, return_index=False, use_cropping=False, RAM=False):
    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames
    self.return_index = return_index
    self.use_cropping = use_cropping
    self.RAM = RAM
    
    if self.RAM:
      self.inputs = []
      self.labels = []
      for index in range(len(self.list_IDs)):
        if index % 10000 == 0:
          print(f"loading {index}")
        image_f = self.list_IDs[index]
        img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
        img = np.transpose(img, [2, 0, 1])
        green = np.expand_dims(img[1,...], axis=0)
        mosaic = bayer(img)
        mosaic = np.sum(mosaic, axis=0, keepdims=True)

        if self.use_cropping: # take center crop of image
          mosaic = mosaic[...,32:94,32:94]
          green = green[...,32:94,32:94]

        self.inputs.append(mosaic)
        self.labels.append(green)

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    if self.RAM:
      if self.return_index:
        return (index, self.inputs[index], self.labels[index])
      else:
        return (self.inputs[index], self.labels[index])

    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])
    green = np.expand_dims(img[1,...], axis=0)
    mosaic = bayer(img)
    mosaic = np.sum(mosaic, axis=0, keepdims=True)
  
    if self.return_index:
      return (index, mosaic, green)
    if self.use_cropping: # take center crop of image
      mosaic = mosaic[...,32:94,32:94]
      green = green[...,32:94,32:94]
    return (mosaic, green) 

  def save_kcore_filenames(self, kcore_ids, kcore_filename):
    with open(kcore_filename, "w") as f:
      for kcore_id in kcore_ids:
        f.write(self.list_IDs[kcore_id] + "\n")
    
