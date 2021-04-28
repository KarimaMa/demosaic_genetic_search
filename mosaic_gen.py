import numpy as np
from torch import nn
import torch

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


def xtrans_cell(h=None, w=None):
  g_pos = [(0,0),        (0,2), (0,3),        (0,5),
                  (1,1),               (1,4),
           (2,0),        (2,2), (2,3),        (2,5),
           (3,0),        (3,2), (3,3),        (3,5),
                  (4,1),               (4,4),
           (5,0),        (5,2), (5,3),        (5,5)]
  r_pos = [(0,4),
           (1,0), (1,2),
           (2,4),
           (3,1),
           (4,3), (4,5),
           (5,1)]
  b_pos = [(0,1),
           (1,3), (1,5),
           (2,1),
           (3,4),
           (4,0), (4,2),
           (5,4)]

  mask = np.zeros((3, 6, 6), dtype=np.float32)

  for idx, coord in enumerate([r_pos, g_pos, b_pos]):
    for y, x in coord:
      mask[..., idx, y, x] = 1

  if h is None or w is None:
    return mask

  h = int(h)
  w = int(w)

  new_sz = [np.ceil(h / 6).astype(np.int32), np.ceil(w / 6).astype(np.int32)]

  sz = np.array(mask.shape)
  sz[:-2] = 1
  sz[-2:] = new_sz
  sz = list(sz)

  mask = np.tile(mask, sz)

  return mask


def xtrans(im, mask=None):
  """XTrans Mosaick.

   The patterned assumed is::

     G b G G r G
     r G r b G b
     G b G G r G
     G r G G b G
     b G b r G r
     G r G G b G

  Args:
    im(np.array, th.Tensor): image to mosaic. Dimensions are [c, h, w]
    mask(bool): if true return the binary mosaic mask, instead of the mosaic image.

  Returns:
    np.array: mosaicked image (if mask==False), or binary mask if (mask==True)
  """

  if not mask is None:
    return mask * im

  mask = xtrans_cell()
 
  h, w = im.shape[-2:]
  h = int(h)
  w = int(w)

  new_sz = [np.ceil(h / 6).astype(np.int32), np.ceil(w / 6).astype(np.int32)]

  sz = np.array(mask.shape)
  sz[:-2] = 1
  sz[-2:] = new_sz
  sz = list(sz)

  mask = np.tile(mask, sz)

  return mask*im



def unshuffle(x, factor):
  outshape = list(x.shape)
  outshape[0] = factor**2
  outshape[1] //= 6
  outshape[2] //= 6

  out = torch.empty(outshape)
  for i in range(factor):
    for j in range(factor):
      c = i * factor + j
      out[c,:,:] = x[0, i::factor, j::factor] 

  return out


"""
packs every 6x6 pixels in 3x3 grid order
"""
def pack3x3(input):
  period = 6
  outshape = list(input.shape)
  outshape[0] = period**2
  outshape[1] //= period
  outshape[2] //= period

  out = np.zeros(outshape, dtype=np.float32)
  num_blocks = 4
  block_size = 9

  for b in range(num_blocks):
    for i in range(block_size):
      bx = b % 2
      by = b // 2
      x = bx * 3 + i % 3
      y = by * 3 + i // 3
      c = b * block_size + i
      out[c, :, :] = input[0, y::period, x::period]
 
  return out


"""
returns the spatially invarian† xtrans mosaic
"""
def xtrans_invariant(im):
  m_size = list(im.shape)
  m_size[0] = 36
  m_size[1] //= 6
  m_size[2] //= 6

  mosaic = xtrans(im)

  mosaic = np.sum(mosaic, axis=0, keepdims=True)
  mosaic = torch.Tensor(mosaic)
  #pixel_unshuffle = nn.PixelUnshuffle(6)
  packed = unshuffle(mosaic, 6)
  return packed


"""
takes flat 1 channel xtrans mosaic
returns the spatially invarian† xtrans mosaic
packed in 3x3 grid format
"""
def xtrans_3x3_invariant(flat_xtrans):
  packed = pack3x3(flat_xtrans)
  return packed


if __name__ == "__main__":

  import argparse
  import os
  from imageio import imread

  parser = argparse.ArgumentParser()
  parser.add_argument("--outdir", type=str)
  parser.add_argument("--filelist", type=str)
  parser.add_argument("--xtrans", action='store_true')
  parser.add_argument("--flat", action='store_true')

  args = parser.parse_args()
  if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
    
  images = [l.strip() for l in open(args.filelist)]

  if args.xtrans:
    mask = xtrans_cell(h=120, w=120)

  for image_f in images:
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])
    # cutoff one pixel from the right left / top bottom to make image w, h a multiple of 6
    img = img[:,4:-4,4:-4]

    if args.xtrans:
      if args.flat:
        mosaic = xtrans(img, mask=mask)
      else:
        mosaic = xtrans_invariant(img)
        mosaic = mosaic.numpy()
    else:
      raise NotImplmentedError 

    image_id = image_f.split("/")[-1].replace('.png', '')
    mosaic_dir = os.path.join(args.outdir, "/".join(image_f.split("/")[-4:-1]))
    if not os.path.exists(mosaic_dir):
      os.makedirs(mosaic_dir)

    mosaic_f = os.path.join(mosaic_dir, image_id + "_mosaic")

    mosaic.astype('float32').tofile(mosaic_f)












