import torch
import numpy as np
from imageio import imread


"""
returns the largest number <= value that is 
a multiple of factor
"""
def get_largest_multiple(value, factor):
  for i in range(value, 0, -1):
    if i % factor == 0 and (i / 2) % 2 == 0:
      return i

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
takes flat 1 channel xtrans mosaic
returns the spatially invarian† xtrans mosaic
packed in 3x3 grid format
"""
def xtrans_3x3_invariant(flat_xtrans):
  packed = pack3x3(flat_xtrans)
  return packed


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


def get_xtrans_input(image_f):
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])
    oldh = img.shape[-2]
    oldw = img.shape[-1]

    h = get_largest_multiple(oldh, 6)
    w = get_largest_multiple(oldw, 6)
    hc = (oldh - h)//2
    wc = (oldw - w)//2

    if hc != 0:
      img = img[:,hc:-hc,:]
    if wc != 0:
      img = img[:,:,wc:-wc]

    mask = xtrans_cell(h=img.shape[-2], w=img.shape[-1])
    mosaic3chan = xtrans(img, mask=mask)

    flat_mosaic = np.sum(mosaic3chan, axis=0, keepdims=True)
    packed_mosaic = xtrans_3x3_invariant(flat_mosaic)

    # extract out the RB values from the mosaic
    period = 6
    num_blocks = 4
    rb_shape = list(flat_mosaic.shape)
    rb_shape[0] = 16
    rb_shape[1] //= period
    rb_shape[2] //= period

    rb = np.zeros(rb_shape, dtype=np.float32)
    num_blocks = 4
    block_size = 4 # 4 red and blue values per 3x3

    for b in range(num_blocks):
      for i in range(block_size):
        bx = b % 2
        by = b // 2
        x = bx * 3 + (i*2+1) % 3
        y = by * 3 + (i*2+1) // 3
        c = b * block_size + i
        rb[c, :, :] = flat_mosaic[0, y::period, x::period]
 
    rb = torch.Tensor(rb)
    flat_mosaic = torch.Tensor(flat_mosaic).unsqueeze(0)
    mosaic3chan = torch.Tensor(mosaic3chan).unsqueeze(0)
    packed_mosaic = torch.Tensor(packed_mosaic).unsqueeze(0)

    input = mosaic3chan, packed_mosaic, flat_mosaic, rb
    
    target = torch.Tensor(img)
  
    return (input, target) 
 

 