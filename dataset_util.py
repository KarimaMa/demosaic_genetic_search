import math
import torch
import os
import numpy as np


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
crops images to size that can be properly handled by bayer 
quad resolution and downsampling / upsampling operators
GIVEN a minimum resolution size of 1/6 
ONLY WORKS FOR SCALE FACTORS OF 2
"""
def get_cropped_img_size(img_w):
  k = math.floor((img_w-6)/2) - 1
  while k % 12 != 0:
    k = k-1
  # print(f"chosen lowres width {k}")
  cropped_size = 2 * (k + 1) + 6
  return cropped_size

def ids_from_file(filename):
  # Assume the list file has relative paths
  root = os.path.dirname(os.path.abspath(filename))
  # root = os.getcwd()
  ids = [os.path.join(root, l.strip()) for l in open(filename, "r")]
  if not os.path.exists(ids[0]):
      raise RuntimeError(f"Dataset filelist is invalid, coult not find {ids[0]}")
  return ids

def get_largest_multiple(value, factor):
  for i in range(value, 0, -1):
    if i % factor == 0 and (i / 2) % 2 == 0:
      return i


def get_image_id(image_f):
  subdir = "/".join(image_f.split("/")[-3:-1])
  image_id = image_f.split("/")[-1]
  return subdir, image_id




def tensor2image(t, normalize=False, dtype=np.uint8):
    """Converts an tensor image (4D tensor) to a numpy 8-bit array.

    Args:
        t(th.Tensor): input tensor with dimensions [bs, c, h, w], c=3, bs=1
        normalize(bool): if True, normalize the tensor's range to [0, 1] before
            clipping
    Returns:
        (np.array): [h, w, c] image in uint8 format, with c=3
    """
    assert len(t.shape) == 4, "expected 4D tensor, got %d dimensions" % len(t.shape)
    bs, c, h, w = t.shape

    assert bs == 1, "expected batch_size 1 tensor, got %d" % bs
    t = t.squeeze(0)

    assert c == 3 or c == 1, "expected tensor with 1 or 3 channels, got %d" % c

    if normalize:
        m = t.min()
        M = t.max()
        t = (t-m) / (M-m+1e-8)

    t = torch.clamp(t.permute(1, 2, 0), 0, 1).cpu().detach().numpy()

    if dtype == np.uint8:
        return (255.0*t).astype(np.uint8)
    elif dtype == np.uint16:
        return ((2**16-1)*t).astype(np.uint16)
    else:
        raise ValueError("dtype %s not recognized" % dtype)


def im2tensor(im):
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    elif im.dtype == np.uint16:
        im = im.astype(np.float32) / (2**16-1.0)
    else:
        raise ValueError(f"unknown input type {im.dtype}")
    im = torch.from_numpy(im)
    if len(im.shape) == 2:  # grayscale -> rgb
        im = im.unsqueeze(-1).repeat(1, 1, 3)
    im = im.float().permute(2, 0, 1)
    return im

