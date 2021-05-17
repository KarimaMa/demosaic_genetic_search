import numpy as np
from torch import nn
import torch
import math



"""
bicubic downsampling kernel
returns the weight for a given coordinate relative to the 
interpolation center
"""
def bicubic_weight(x, scale):
  absx = abs(x / scale)
  absx2 = absx**2
  absx3 = absx2 * absx
  if absx <= 1:
    return 1.5 * absx3 - 2.5 * absx2 + 1
  elif absx < 2:
    return -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
  else:
    return 0

"""
assumes scale factors are always integers
"""
def compute_kernel(kwidth, scale_factor):
  kernel = np.zeros(kwidth*2+1)
  for tap in range(-kwidth, kwidth+1):
    if scale_factor % 2 == 0:
      weight = bicubic_weight(tap + 0.5, scale_factor)
    else:
      weight = bicubic_weight(tap, scale_factor)
    kernel[tap+kwidth] = weight
  return kernel


# def bicubic_downsample(image, scale_factor):
#   radius = scale_factor * 2 # for antialiasing 
 
#   # add two additional dimensions for input and output channels
#   kernel = compute_kernel(radius, scale_factor)
#   kernel_sum = np.sum(kernel)
#   kernel_x = np.expand_dims(kernel, axis=0)
#   kernel_y = np.expand_dims(kernel, axis=1)

#   # add channel dims 
#   kernel_x = np.tile(kernel_x, (3,1,1,1))
#   kernel_y = np.tile(kernel_y, (3,1,1,1))

#   kernel_x = torch.Tensor(kernel_x)
#   kernel_y = torch.Tensor(kernel_y)

#   # blur in the x direction

#   resized_x = torch.nn.functional.conv2d(image, kernel_x, groups=3, stride=(1,scale_factor), padding=(0,radius))
#   # normalize resized_x
#   normed_resized_x = resized_x / kernel_sum 

#   resized_y = torch.nn.functional.conv2d(normed_resized_x, kernel_y, groups=3, stride=(scale_factor,1), padding=(radius, 0)) 
#   normed_resized_y = resized_y / kernel_sum
#   return normed_resized_y


def bicubic_downsample(image, factor=None, pad=0):
  radius = factor * 2 # for antialiasing 
 
  # add two additional dimensions for input and output channels
  kernel = compute_kernel(radius, factor)
  kernel_sum = np.sum(kernel)
  kernel_x = np.expand_dims(kernel, axis=0)
  kernel_y = np.expand_dims(kernel, axis=1)

  # add channel dims 
  kernel_x = np.tile(kernel_x, (3,1,1,1))
  kernel_y = np.tile(kernel_y, (3,1,1,1))

  kernel_x = torch.Tensor(kernel_x)
  kernel_y = torch.Tensor(kernel_y)

  # blur in the x direction

  #resized_x = torch.nn.functional.conv2d(image, kernel_x, groups=3, stride=(1,scale_factor))
  resized_x = torch.nn.functional.conv2d(image, kernel_x, groups=3, stride=(1,factor), padding=(0,pad))

  # normalize resized_x
  normed_resized_x = resized_x / kernel_sum 

  resized_y = torch.nn.functional.conv2d(normed_resized_x, kernel_y, groups=3, stride=(factor,1), padding=(pad,0)) 
  normed_resized_y = resized_y / kernel_sum
  return torch.clamp(normed_resized_y, 0, 1)


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
  
  #lowres_im = nn.functional.interpolate(im, scale_factor=0.5, mode='bicubic', align_corners=False)
  im = torch.Tensor(im).unsqueeze(0) # add a batch dimension
  lowres_im = bicubic_downsample(im, 2)

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


def downsample_image(im, scale_factor):
  im = torch.Tensor(im).unsqueeze(0) # add a batch dimension
  lowres_im = bicubic_downsample(im, scale_factor)

  return lowres_im



"""
test bicubic downsampling 
"""

if __name__ == "__main__":

  import argparse
  import os
  from imageio import imread
  from PIL import Image

  parser = argparse.ArgumentParser()
  # parser.add_argument("--outdir", type=str)
  parser.add_argument("--filelist", type=str)
  parser.add_argument("--scale_factor", type=int)
  parser.add_argument("--test", action='store_true')
  parser.add_argument("--outdir", type=str)

  args = parser.parse_args()
  scale = args.scale_factor

  images = [l.strip() for l in open(args.filelist)]

  if args.test:
    image_f = np.random.choice(images)
    img = np.array(imread(image_f)).astype(np.float32) / (2**8 - 1)
    img = np.transpose(img, [2, 0, 1])

    input_w = img.shape[-1]
    print(f"original image size: {img.shape}")

    lowres_im = torch.clamp(downsample_image(img, scale), 0, 1) * (2**8 - 1)
    # get rid of batch dimension
    lowres_im = lowres_im[0].numpy()
    print(f"lowres image shape {lowres_im.shape}")

    # crop out invalid parts of lowres image
    kernel_w = scale * 2
    min_out = int(kernel_w / scale)
    max_out = int((input_w - kernel_w) / scale)

    min_in = int(min_out * scale - math.floor(scale/2))
    max_in = int(max_out * scale + math.ceil(scale/2) - 1)

    print(f"min output coord: {min_out} max output coord: {max_out}")
    print(f"min input coord: {min_in} max input coord: {max_in}")

    lowres_im = lowres_im[:,min_out:max_out+1, min_out:max_out+1]
    cropped_fullres = img[:,min_in:max_in+1, min_in:max_in+1] * (2**8 - 1)

    print(f"cropped lowres shape: {lowres_im.shape}")
    print(f"cropped fullres shape: {cropped_fullres.shape}")

    image_id = image_f.split("/")[-1].replace('.png', '')
    lowres_f = os.path.join(image_id + "lowres.png")
    fullres_f = os.path.join(image_id + "fullres.png")

    pil_lowres_img = np.transpose(lowres_im, [1,2,0]).astype(np.uint8)
    pil_fullres_img = np.transpose(cropped_fullres, [1,2,0]).astype(np.uint8)

    Image.fromarray(pil_lowres_img).save(lowres_f)
    Image.fromarray(pil_fullres_img).save(fullres_f)

    print(lowres_f)
    print(fullres_f)
  else:
    for image_f in images:
      if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

      img = np.array(imread(image_f)).astype(np.float32) / (2**8 - 1)
      img = np.transpose(img, [2, 0, 1])
      input_w = img.shape[-1]

      lowres_im = torch.clamp(downsample_image(img, scale), 0, 1) * (2**8 - 1)
      # get rid of batch dimension
      lowres_im = lowres_im[0].numpy()

      # compute valid parts of the lowres and corresponding full res image
      kernel_w = scale * 2
      # coordinate range in output image
      min_out = int(kernel_w / scale)
      max_out = int((input_w - kernel_w) / scale)
      # coordinate range in input image
      min_in = int(min_out * scale - math.floor(scale/2))
      max_in = int(max_out * scale + math.ceil(scale/2) - 1)

      lowres_im = lowres_im[:,min_out:max_out+1, min_out:max_out+1]
      cropped_fullres = img[:,min_in:max_in+1, min_in:max_in+1] * (2**8 - 1)

      pil_lowres_img = np.transpose(lowres_im, [1,2,0]).astype(np.uint8)
      pil_fullres_img = np.transpose(cropped_fullres, [1,2,0]).astype(np.uint8)

      image_id = image_f.split("/")[-1].replace('.png', '')
      datadir = os.path.join(args.outdir, "/".join(image_f.split("/")[-4:-1]))
      if not os.path.exists(datadir):
        os.makedirs(datadir)

      lowres_f = os.path.join(datadir, os.path.join(image_id + "lowres.png"))
      fullres_f = os.path.join(datadir, os.path.join(image_id + "fullres.png"))

      Image.fromarray(pil_lowres_img).save(lowres_f)
      Image.fromarray(pil_fullres_img).save(fullres_f)










