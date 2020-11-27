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
  random.shuffle(ids)
  return ids



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
  def __init__(self, data_file=None, data_filenames=None, return_index=False, RAM=False, flatten=True):
    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames
    self.return_index = return_index
    self.RAM = RAM
    self.flatten = flatten

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
    
    if self.flatten:
      mosaic = np.sum(mosaic, axis=0, keepdims=True)
  
    if self.return_index:
      return (index, mosaic, green)
  
    return (mosaic, green) 

  def save_kcore_filenames(self, kcore_ids, kcore_filename):
    with open(kcore_filename, "w") as f:
      for kcore_id in kcore_ids:
        f.write(self.list_IDs[kcore_id] + "\n")
    

class Dataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None,\
              return_index=False, RAM=False, flatten=True, \
              green_file=None, green_filenames=None, \
              green_input=False, green_pred_input=False, green_output=True):
    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames

    self.green_pred_input = green_pred_input

    if self.green_pred_input:
      assert(not (green_file is None and green_filenames is None)), "missing precomputed green data"
      if green_file:
        self.green_list_IDs = ids_from_file(green_file)
      else:
        self.green_list_IDs = green_filenames

    self.return_index = return_index
    self.RAM = RAM
    self.flatten = flatten
    self.green_output = green_output
    self.green_input = green_input

    if self.RAM:
      self.inputs = []
      self.labels = []
      for index in range(len(self.list_IDs)):
        if index % 10000 == 0:
          print(f"loading {index}")
        image_f = self.list_IDs[index]
        img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
        img = np.transpose(img, [2, 0, 1])

        if self.green_output:
          target = np.expand_dims(img[1,...], axis=0)
        else:
          target = img

        mosaic = bayer(img)

        if self.flatten:
          mosaic = np.sum(mosaic, axis=0, keepdims=True)

        if self.green_input:
          if self.green_pred_input:
            green_f = self.green_list_IDs[index]
            green = np.load(green_f)
          else:
            green = np.expand_dims(img[1,...], axis=0)
          input = (mosaic, green)
        else:
          input = mosaic

        self.inputs.append(input)
        self.labels.append(target)


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

    if self.green_output:
      target = np.expand_dims(img[1,...], axis=0)
    else:
      target = img

    mosaic = bayer(img)
    
    if self.flatten:
      mosaic = np.sum(mosaic, axis=0, keepdims=True)

    if self.green_input:
      if self.green_pred_input:
        green_f = self.green_list_IDs[index]
        green = np.load(green_f)
      else:
        green = np.expand_dims(img[1,...], axis=0)
      input = (mosaic, green)
    else:
      input = mosaic

    if self.return_index:
      return (index, input, target)
  
    return (input, target) 


class RedDataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None,\
              return_index=False, \
              green_file=None, green_filenames=None, \
              green_input=False, green_pred_input=False,\
              redgb=False, redgr=False, redb=False, all_red=False):
    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames

    self.green_pred_input = green_pred_input

    if self.green_pred_input:
      assert(not (green_file is None and green_filenames is None)), "missing precomputed green data"
      if green_file:
        self.green_list_IDs = ids_from_file(green_file)
      else:
        self.green_list_IDs = green_filenames

    self.return_index = return_index
    self.green_input = green_input
    self.redgb = redgb
    self.redgr = redgr
    self.redb = redb
    self.all_red = all_red

    self.mask = np.zeros((1,128 ,128))

    if self.redgr:
      self.mask[0,0::2,0::2] = 1
    elif self.redgb:
      self.mask[0,1::2,1::2] = 1
    elif self.redb:
      self.mask[0,1::2,0::2] = 1

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])

    mosaic = bayer(img)
    mosaic = np.sum(mosaic, axis=0, keepdims=True)

    if self.green_input:
      if self.green_pred_input:
        green_f = self.green_list_IDs[index]
        green = np.load(green_f)
      else:
        green = np.expand_dims(img[1,...], axis=0)

      quad_size = list(img.shape)
      quad_size[0] = 6
      quad_size[1] //= 2
      quad_size[2] //= 2

      quad = np.zeros(quad_size)
      quad[0,:,:] = mosaic[0,0::2,0::2]
      quad[1,:,:] = mosaic[0,0::2,1::2]
      quad[2,:,:] = mosaic[0,1::2,0::2]
      quad[3,:,:] = mosaic[0,1::2,1::2]
      quad[4,:,:] =  green[0,0::2,1::2]
      quad[5,:,:] =  green[0,1::2,0::2]

    else:
      quad_size = list(img.shape)
      quad_size[0] = 4
      quad_size[1] //= 2
      quad_size[2] //= 2

      quad = np.zeros(quad_size)
      quad[0,:,:] = mosaic[0,0::2,0::2]
      quad[1,:,:] = mosaic[0,0::2,1::2]
      quad[2,:,:] = mosaic[0,1::2,0::2]
      quad[3,:,:] = mosaic[0,1::2,1::2]
    
    input = green, mosaic, img

    if self.all_red:
      target = np.expand_dims(img[0,...], axis=0)
    else:
      target = np.expand_dims(img[0,...], axis=0) * self.mask # red

    if self.return_index:
      return (index, input, target)
  
    return (input, target) 
 

class RedQuadDataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None,\
              return_index=False, \
              green_file=None, green_filenames=None, \
              green_input=False, green_pred_input=False):
    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames

    self.green_pred_input = green_pred_input

    if self.green_pred_input:
      assert(not (green_file is None and green_filenames is None)), "missing precomputed green data"
      if green_file:
        self.green_list_IDs = ids_from_file(green_file)
      else:
        self.green_list_IDs = green_filenames

    self.return_index = return_index
    self.green_input = green_input

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])

    mosaic = bayer(img)
    mosaic = np.sum(mosaic, axis=0, keepdims=True)

    quad_size = list(img.shape)
    quad_size[0] = 4
    quad_size[1] //= 2
    quad_size[2] //= 2

    if self.green_input:
      if self.green_pred_input:
        green_f = self.green_list_IDs[index]
        green = np.load(green_f)
      else:
        green = np.expand_dims(img[1,...], axis=0)

      green_quad = np.zeros(quad_size)
      green_quad[0,:,:] = green[0,0::2,0::2]
      green_quad[1,:,:] = green[0,0::2,1::2]
      green_quad[2,:,:] = green[0,1::2,0::2]
      green_quad[3,:,:] = green[0,1::2,1::2]

    bayer_quad = np.zeros(quad_size)
    bayer_quad[0,:,:] = mosaic[0,0::2,0::2]
    bayer_quad[1,:,:] = mosaic[0,0::2,1::2]
    bayer_quad[2,:,:] = mosaic[0,1::2,0::2]
    bayer_quad[3,:,:] = mosaic[0,1::2,1::2]

    if self.green_input:
      input = green_quad, bayer_quad, img
    else:
      input = bayer_quad, img

    target = np.expand_dims(img[0,...], axis=0) # red

    if self.return_index:
      return (index, input, target)
  
    return (input, target) 
 

class BlueDataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None,\
              return_index=False, \
              green_file=None, green_filenames=None, \
              green_input=False, green_pred_input=False,\
              bluegb=False, bluegr=False, bluer=False, all_blue=False):
    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames

    self.green_pred_input = green_pred_input

    if self.green_pred_input:
      assert(not (green_file is None and green_filenames is None)), "missing precomputed green data"
      if green_file:
        self.green_list_IDs = ids_from_file(green_file)
      else:
        self.green_list_IDs = green_filenames

    self.return_index = return_index
    self.green_input = green_input
    self.bluegb = bluegb
    self.bluegr = bluegr
    self.bluer = bluer
    self.all_blue = all_blue

    self.mask = np.zeros((1,128 ,128))

    if self.bluegr:
      self.mask[0,0::2,0::2] = 1
    elif self.bluegb:
      self.mask[0,1::2,1::2] = 1
    elif self.bluer:
      self.mask[0,0::2,1::2] = 1

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])

    mosaic = bayer(img)
    mosaic = np.sum(mosaic, axis=0, keepdims=True)

    if self.green_input:
      if self.green_pred_input:
        green_f = self.green_list_IDs[index]
        green = np.load(green_f)
      else:
        green = np.expand_dims(img[1,...], axis=0)

    input = green, mosaic, img

    if self.all_blue:
      target = np.expand_dims(img[2,...], axis=0)
    else:
      target = np.expand_dims(img[2,...], axis=0) * self.mask # blue

    if self.return_index:
      return (index, input, target)
  
    return (input, target) 
 

class RedBlueQuadDataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None,\
              return_index=False, \
              green_file=None, green_filenames=None, \
              green_input=False, green_pred_input=False):
    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames

    self.green_pred_input = green_pred_input

    if self.green_pred_input:
      assert(not (green_file is None and green_filenames is None)), "missing precomputed green data"
      if green_file:
        self.green_list_IDs = ids_from_file(green_file)
      else:
        self.green_list_IDs = green_filenames

    self.return_index = return_index
    self.green_input = green_input

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])

    mosaic = bayer(img)
    mosaic = np.sum(mosaic, axis=0, keepdims=True)

    quad_size = list(img.shape)
    quad_size[0] = 4
    quad_size[1] //= 2
    quad_size[2] //= 2

    if self.green_input:
      if self.green_pred_input:
        green_f = self.green_list_IDs[index]
        green = np.load(green_f)
      else:
        green = np.expand_dims(img[1,...], axis=0)

      green_quad = np.zeros(quad_size)
      green_quad[0,:,:] = green[0,0::2,0::2]
      green_quad[1,:,:] = green[0,0::2,1::2]
      green_quad[2,:,:] = green[0,1::2,0::2]
      green_quad[3,:,:] = green[0,1::2,1::2]

    bayer_quad = np.zeros(quad_size)
    bayer_quad[0,:,:] = mosaic[0,0::2,0::2]
    bayer_quad[1,:,:] = mosaic[0,0::2,1::2]
    bayer_quad[2,:,:] = mosaic[0,1::2,0::2]
    bayer_quad[3,:,:] = mosaic[0,1::2,1::2]

    if self.green_input:
      input = green_quad, bayer_quad, img
    else:
      input = bayer_quad, img

    red_target = np.expand_dims(img[0,...], axis=0) # red
    blue_target = np.expand_dims(img[2,...], axis=0) # blue

    target = np.concatenate([red_target, blue_target], axis=0)

    if self.return_index:
      return (index, input, target)
  
    return (input, target) 



class QuadDataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None,\
              return_index=False, \
              green_file=None, green_filenames=None, \
              green_input=False, green_pred_input=False):

    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames

    self.green_pred_input = green_pred_input
    self.green_input = green_input

    if green_input:
      if self.green_pred_input:
        assert(not (green_file is None and green_filenames is None)), "missing precomputed green data"
        if green_file:
          self.green_list_IDs = ids_from_file(green_file)
        else:
          self.green_list_IDs = green_filenames

    self.return_index = return_index

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])

    mosaic = bayer(img)
    mosaic = np.sum(mosaic, axis=0, keepdims=True)

    quad_size = list(img.shape)
    quad_size[0] = 4
    quad_size[1] //= 2
    quad_size[2] //= 2

    if self.green_input:
      if self.green_pred_input:
        green_f = self.green_list_IDs[index]
        green = np.load(green_f)
      else:
        green = np.expand_dims(img[1,...], axis=0)

      green_quad = np.zeros(quad_size)
      green_quad[0,:,:] = green[0,0::2,0::2]
      green_quad[1,:,:] = green[0,0::2,1::2]
      green_quad[2,:,:] = green[0,1::2,0::2]
      green_quad[3,:,:] = green[0,1::2,1::2]

    bayer_quad = np.zeros(quad_size)
    bayer_quad[0,:,:] = mosaic[0,0::2,0::2]
    bayer_quad[1,:,:] = mosaic[0,0::2,1::2]
    bayer_quad[2,:,:] = mosaic[0,1::2,0::2]
    bayer_quad[3,:,:] = mosaic[0,1::2,1::2]

    if self.green_input:
      input = (green_quad, bayer_quad)
    else:
      input = bayer_quad

    target = img

    if self.return_index:
      return (index, input, target)

    return (input, target) 
 

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

    mosaic = bayer(img)
    mosaic = np.sum(mosaic, axis=0, keepdims=True)

    quad_size = list(img.shape)
    quad_size[0] = 4
    quad_size[1] //= 2
    quad_size[2] //= 2

    bayer_quad = np.zeros(quad_size)
    bayer_quad[0,:,:] = mosaic[0,0::2,0::2]
    bayer_quad[1,:,:] = mosaic[0,0::2,1::2]
    bayer_quad[2,:,:] = mosaic[0,1::2,0::2]
    bayer_quad[3,:,:] = mosaic[0,1::2,1::2]

    input = bayer_quad

    target = np.expand_dims(img[1,...], axis=0)

    if self.return_index:
      return (index, input, target)
  
    return (input, target) 
 

 