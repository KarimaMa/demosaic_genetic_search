import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data
from imageio import imread, imsave
from PIL import Image
from config import IMG_H, IMG_W
from dataset_util import ids_from_file, get_cropped_img_size, tensor2image
from superres_mosaic_gen import BicubicDownsample, bicubic_downsample
import math
import sys
sr_baseline_dir = os.path.join("/".join(sys.path[0].split("/")[0:3]), "sr-baselines")
print(sr_baseline_dir)
sys.path.append(sr_baseline_dir)
from upsample import BicubicUpsample
from downsample import GaussianDownsample


class SRCNNDataset(data.Dataset):
  def __init__(self, data_file=None, return_index=False):
    
    if data_file:
        self.list_IDs = ids_from_file(data_file)
    else: 
        self.input_IDs = ids_from_file(input_data_file) # patch filenames
        self.target_IDs = ids_from_file(target_data_file)

    self.return_index = return_index
    self.upsampler = BicubicUpsample(3)
    self.downsampler = GaussianDownsample()

  def __len__(self):
    return len(self.list_IDs)

  def get_start_end_locs(self, w):
    SCALE = 2
    kw = SCALE * 2 # for antialiasing
    # fist input pixel that we should center our convolution taps on to produce
    # the dowsnampled image. This is computed by normalizing the innput and output images 
    # and finding the closest pixel in the input that corresponds to the center of the first output pixel 
    first_input_loc = math.floor(0.5 * SCALE)
    # the padding we need to add to force pytorch to start convolving at the desired location
    pad = kw - first_input_loc
    # size of the downsampled image produced by pytorch BEFORE propper cropping
    downsampled_size = (w + 2*pad - (2*kw+1)) // SCALE + 1
       
    # the relationship between convolution center pixels in the input with pixels in the output is: 
    #  1) input_loc = output_loc * SCALE + first_input_loc 
    # The first valid center pixel in the input that we can ask the model to predict from the downsampled image 
    # must correspond to the first output pixel whose convolution support window is within the input image bounds:
    #  2) input_loc >= kw to produce valid output loc
    # Combining the two equations we have: first_valid_output_loc * SCALE + first_input_loc >= kw
    first_valid_out_loc = math.ceil( (kw - first_input_loc) / SCALE ) 
    # Similarly, the right most center pixel in the input that we can ask the model to predict must correspond to the 
    # rightmost output pixel whose convolution support window is within the input image
    #  3) input_loc <= (w-1) - kw to produce valid output loc    
    # Combining the two equatios we have: last_valid_output_loc * SCALE + first_input_loc <= w-1 - kw
    last_valid_out_loc = math.floor(((w-1 - kw) - first_input_loc) / SCALE) 

    # use equation 1 to compute the corresponding leftmost and rightmost input pixel convolution centers
    first_valid_in_loc = first_valid_out_loc * SCALE + first_input_loc
    last_valid_in_loc = last_valid_out_loc * SCALE + first_input_loc

    # Expand the bounds on the input by centering a window with size equal to the downsampling factor 
    # on the leftmost and right most valid input convolution centers
    # For even scale factors, we center the window between the center pixel and the pixel above or to its left 
    # because for even scale factors, the true center of the convolution lies between input pixels 
    # so the discrete center pixel is chosen to be the pixel to the right / below the actual center location 
    first_valid_in_loc -= (SCALE // 2)
    last_valid_in_loc += int((SCALE - 0.5) // 2) 

    return (first_valid_out_loc, last_valid_out_loc), (first_valid_in_loc, last_valid_in_loc), pad

  def __getitem__(self, index):  
    img = Image.open(self.list_IDs[index]).convert("RGB")
    try:
        img = np.array(img).astype(np.float32)
    except TypeError as e:
        print(image_f, e)

    if len(img.shape) == 2: #grayscale image
        img = np.expand_dims(img, 2)
        img = np.tile(img, (1,1,3))
    
    img = np.transpose(img, [2, 0, 1])

    channels, height, width = img.shape
    dim = 128

    l = np.random.randint(0, width-dim+1)
    t = np.random.randint(0, height-dim+1)

    # take a random crop of the image
    img = img[:,t:t+dim,l:l+dim]
    img = torch.from_numpy(np.array(img)).float() / 255.0
    
    h = img.shape[-2]
    w = img.shape[-1]
    # crop image so that the downsampled image h, w will be a multiple of 12
    new_h = get_cropped_img_size(h)
    new_w = get_cropped_img_size(w)

    hcrop = h - new_h
    wcrop = w - new_w

    img = img[:, hcrop:, wcrop:]

    (hlowres_start, hlowres_end), (hfullres_start, hfullres_end), pad = self.get_start_end_locs(img.shape[-2])
    (wlowres_start, wlowres_end), (wfullres_start, wfullres_end), pad = self.get_start_end_locs(img.shape[-1])

    # add a batch dimension to the image to use torch convolutions for downsampling
    batched_img = torch.Tensor(img).unsqueeze(0)
    #lowres_img = self.downsampler(batched_img)
    lowres_img = bicubic_downsample(batched_img, factor=2, pad=pad)[0]

    # crop out the valid region of the lowres image
    lowres_img = lowres_img[:,hlowres_start:hlowres_end+1, wlowres_start:wlowres_end+1]
    upsampled = self.upsampler(lowres_img)
    target = img
    # crop out the valid region of the fullres image 
    target = target[:,hfullres_start:hfullres_end+1, wfullres_start:wfullres_end+1]
    target = torch.Tensor(target)

    input = upsampled

    if self.return_index:
        return (index, input, target)  
    else:
        return (input, target)
