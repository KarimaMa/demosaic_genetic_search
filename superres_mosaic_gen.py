import numpy as np
from torch import nn
import torch



"""
bicubic downsampling kernel
returns the weight for a given coordinate relative to the 
interpolation center
"""
def bicubic_weight(x, scale):
  absx = math.abs(x / scale)
  absx2 = absx**2
  absx3 = absx2 * absx
  if absx < 1:
    return 1.5 * absx3 - 2.5 * absx3 + 1
  elif absx < 2:
    return -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
  else:
    return 0



"""
from mgharbi demosaicnet code
"""

"""
returns low-res bayer mosaic
"""
def lowres_bayer(im, return_mask=False):
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
  lowres_im = nn.functional.interpolate(im, scale_factor=0.5, mode='bicubic', align_corners=False)
  lowres_im = lowres_im[0]
  mask = np.ones_like(lowres_im)

  # red
  mask[0, ::2, 0::2] = 0
  mask[0, 1::2, :] = 0

  # green
  mask[1, ::2, 1::2] = 0
  mask[1, 1::2, ::2] = 0

  # blue
  mask[2, 0::2, :] = 0
  mask[2, 1::2, 1::2] = 0

  if return_mask:
    return mask

  return lowres_im*mask
