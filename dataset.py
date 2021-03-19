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
  # Assume the list file has relative paths
  root = os.getcwd()
  ids = [os.path.join(root, l.strip()) for l in open(filename, "r")]
  random.shuffle(ids)
  if not os.path.exists(ids[0]):
      raise RuntimeError(f"Dataset filelist is invalid, coult not find {ids[0]}")
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
              return_index=False, flatten=True, \
              green_file=None, green_filenames=None, green_input=False):

    if data_file:
      self.list_IDs = ids_from_file(data_file) # patch filenames
    else:
      self.list_IDs = data_filenames

    if green_file:
      self.precomputed_green = True
      self.green_list_IDs = ids_from_file(green_file)
    elif green_filenames:
      self.precomputed_green = True
      self.green_list_IDs = green_filenames

    self.return_index = return_index
    self.flatten = flatten
    self.green_input = green_input

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])

    target = img
    mosaic = bayer(img)
    
    if self.flatten:
      mosaic = np.sum(mosaic, axis=0, keepdims=True)

    if self.green_input:
      if self.precomputed_green:
        green_f = self.green_list_IDs[index]
        green = np.load(green_f) # precomputed predicted green 
      else:
        green = np.expand_dims(img[1,...], axis=0) # use ground truth green 
      input = (mosaic, green)
    else:
      input = mosaic

    if self.return_index:
      return (image_f, input, target)
  
    return (input, target) 


class GradHalideDataset(data.Dataset):
  def __init__(self, data_file=None, data_filenames=None,\
              return_index=False):

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

    target = img
    mosaic = bayer(img)
    
    flat_mosaic = np.sum(mosaic, axis=0, keepdims=True)

    input = (flat_mosaic, mosaic)

    if self.return_index:
      return (image_f, input, target)
  
    return (input, target) 



"""
provides bayer quad, red and blue from bayer, gr and gb from bayer
as inputs and the full RGB image as the target
"""
class FullPredictionQuadDataset(data.Dataset):
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

    image_size = list(img.shape)
    quad_h = image_size[1] // 2
    quad_w = image_size[2] // 2

    redblue_bayer = np.zeros((2, quad_h, quad_w))
    bayer_quad = np.zeros((4, quad_h, quad_w))
    green_grgb = np.zeros((2, quad_h, quad_w))

    bayer_quad[0,:,:] = mosaic[0,0::2,0::2]
    bayer_quad[1,:,:] = mosaic[0,0::2,1::2]
    bayer_quad[2,:,:] = mosaic[0,1::2,0::2]
    bayer_quad[3,:,:] = mosaic[0,1::2,1::2]

    redblue_bayer[0,:,:] = bayer_quad[1,:,:]
    redblue_bayer[1,:,:] = bayer_quad[2,:,:]

    green_grgb[0,:,:] = bayer_quad[0,:,:]
    green_grgb[1,:,:] = bayer_quad[3,:,:]

    target = img

    input = (bayer_quad, redblue_bayer, green_grgb)

    if self.return_index:
      return (image_f, input, target)

    return (input, target) 
 

"""
provides bayer quad, red and blue from bayer, gr and gb from bayer
as inputs and the full RGB image as the target
"""
class FullPredictionProcessedDataset(data.Dataset):
  def __init__(self, data_file=None, return_index=False):
    self.list_IDs = ids_from_file(data_file) # patch filenames
    self.return_index = return_index

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_datadir = self.list_IDs[index]

    target_f = os.path.join(image_datadir, "rgb_target")
    target = np.fromfile(target_f, dtype=np.float32).reshape(3, 128, 128)

    bayer_f = os.path.join(image_datadir, "bayer")
    bayer_quad = np.fromfile(bayer_f, dtype=np.float32).reshape(4, 64, 64)

    redblue_bayer_f = os.path.join(image_datadir, "redblue_bayer")
    redblue_bayer = np.fromfile(redblue_bayer_f, dtype=np.float32).reshape(2, 64, 64)

    green_grgb_f = os.path.join(image_datadir, "green_grgb")
    green_grgb = np.fromfile(green_grgb_f, dtype=np.float32).reshape(2, 64, 64)

    input = (bayer_quad, redblue_bayer, green_grgb)

    if self.return_index:
      return (image_datadir, input, target)

    return (input, target) 
 

class RGB8ChanDataset(data.Dataset):
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
 

 
class GreenProcessedQuadDataset(data.Dataset):
  def __init__(self, data_file=None, return_index=False):
    self.list_IDs = ids_from_file(data_file) # patch filenames
    self.return_index = return_index

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_datadir = self.list_IDs[index]

    target_f = os.path.join(image_datadir, "green_target")
    target = np.fromfile(target_f, dtype=np.float32).reshape(1, 128, 128)

    bayer_f = os.path.join(image_datadir, "bayer")
    bayer_quad = np.fromfile(bayer_f, dtype=np.float32).reshape(4, 64, 64)

    input = bayer_quad

    if self.return_index:
      return (index, input, target)

    return (input, target) 

