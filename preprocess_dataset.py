import os
from imageio import imread
import numpy as np
import pathlib
import argparse


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
  if not os.path.exists(ids[0]):
      raise RuntimeError(f"Dataset filelist is invalid, coult not find {ids[0]}")
  return ids


def process_images(outdir, image_name_file_list):
  image_names = ids_from_file(image_name_file_list) # patch filenames

  for image_f in image_names:
    dataset = image_f.split("/")[-3]
    image_group = image_f.split("/")[-2]
    image_id = image_f.split("/")[-1].strip(".png")

    image_datadir = os.path.join(outdir, os.path.join(os.path.join(dataset, image_group), image_id))
    pathlib.Path(image_datadir).mkdir(parents=True, exist_ok=True) 
    print(image_group)

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

    rgb_target = img
    green_target = np.expand_dims(img[1,...], axis=0)

    rgb_target_f = os.path.join(image_datadir, "rgb_target")
    rgb_target.astype('float32').tofile(rgb_target_f)

    green_target_f = os.path.join(image_datadir, "green_target")
    green_target.astype('float32').tofile(green_target_f)

    bayer_f = os.path.join(image_datadir, "bayer")
    bayer_quad.astype('float32').tofile(bayer_f)

    rb_bayer_f = os.path.join(image_datadir, "redblue_bayer")
    redblue_bayer.astype('float32').tofile(rb_bayer_f)

    green_grgb_f = os.path.join(image_datadir, "green_grgb")
    green_grgb.astype('float32').tofile(green_grgb_f)

 

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--outdir", type=str, help="directory for processed data")
  parser.add_argument("--filelist", type=str, help="file with list of dataset image filenames")

  args = parser.parse_args()
  process_images(args.outdir, args.filelist)



