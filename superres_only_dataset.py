import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data
from imageio import imread
from config import IMG_H, IMG_W
from util import ids_from_file
from mosaic_gen import bayer
from superres_mosaic_gen import lowres_bayer, bicubic_downsample
import math


class SOnlyGreenQuadDataset(data.Dataset):
  def __init__(self, data_file=None, return_images=False, return_index=False):
    
    if data_file:
        self.list_IDs = ids_from_file(data_file)
    else: 
        self.input_IDs = ids_from_file(input_data_file) # patch filenames
        self.target_IDs = ids_from_file(target_data_file)

    self.return_index = return_index

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])

    SCALE = 2
    kw = SCALE * 2 # for antialiasing
    # fist input pixel that we should center our convolution taps on to produce
    # the dowsnampled image. This is computed by normalizing the innput and output images 
    # and finding the closest pixel in the input that corresponds to the center of the first output pixel 
    first_input_loc = math.floor(0.5 * SCALE)
    # the padding we need to add to force pytorch to start convolving at the desired location
    pad = kw - first_input_loc
    image_w = img.shape[-1]
    # size of the downsampled image produced by pytorch BEFORE propper cropping
    downsampled_size = (image_w + 2*pad - (2*kw+1)) // SCALE + 1
       
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
    last_valid_out_loc = math.floor(((image_w-1 - kw) - first_input_loc) / SCALE) 

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

    # add a batch dimension to the image to use torch convolutions for downsampling
    batched_img = torch.Tensor(img).unsqueeze(0)
    lowres_img = bicubic_downsample(batched_img, factor=2, pad=pad)[0]

    # print(f"using pad {pad} for kernel radius {kw} and scale factor {SCALE}")
    # print(f"computed downsampled size: {downsampled_size}  downsampled_size size before cropping : {lowres_img.shape}")
    
    # crop out the valid region of the lowres image
    lowres_img = lowres_img[:,first_valid_out_loc:last_valid_out_loc+1, first_valid_out_loc:last_valid_out_loc+1]
    
    input = lowres_img
  
    target = np.expand_dims(img[1,...], axis=0)
    # crop out the valid region of the fullres image 
    target = target[:,first_valid_in_loc:last_valid_in_loc+1, first_valid_in_loc:last_valid_in_loc+1]
    
    target = torch.Tensor(target)

    if self.return_index:
        return (index, input, target)  
    else:
        return (input, target)


class SDataset(data.Dataset):
  def __init__(self, data_file=None, return_index=False):
    
    if data_file:
        self.list_IDs = ids_from_file(data_file)
    else: 
        self.input_IDs = ids_from_file(input_data_file) # patch filenames
        self.target_IDs = ids_from_file(target_data_file)

    self.return_index = return_index

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
    img = np.transpose(img, [2, 0, 1])

    SCALE = 2
    kw = SCALE * 2 # for antialiasing
    # fist input pixel that we should center our convolution taps on to produce
    # the dowsnampled image. This is computed by normalizing the innput and output images 
    # and finding the closest pixel in the input that corresponds to the center of the first output pixel 
    first_input_loc = math.floor(0.5 * SCALE)
    # the padding we need to add to force pytorch to start convolving at the desired location
    pad = kw - first_input_loc
    image_w = img.shape[-1]
    # size of the downsampled image produced by pytorch BEFORE propper cropping
    downsampled_size = (image_w + 2*pad - (2*kw+1)) // SCALE + 1
       
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
    last_valid_out_loc = math.floor(((image_w-1 - kw) - first_input_loc) / SCALE) 

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

    # add a batch dimension to the image to use torch convolutions for downsampling
    batched_img = torch.Tensor(img).unsqueeze(0)
    lowres_img = bicubic_downsample(batched_img, factor=2, pad=pad)[0]

    # print(f"using pad {pad} for kernel radius {kw} and scale factor {SCALE}")
    # print(f"computed downsampled size: {downsampled_size}  downsampled_size size before cropping : {lowres_img.shape}")
    
    # crop out the valid region of the lowres image
    lowres_img = lowres_img[:,first_valid_out_loc:last_valid_out_loc+1, first_valid_out_loc:last_valid_out_loc+1]

    target = img
    # crop out the valid region of the fullres image 
    target = target[:,first_valid_in_loc:last_valid_in_loc+1, first_valid_in_loc:last_valid_in_loc+1]

    target = torch.Tensor(target)

    input = lowres_img

    if self.return_index:
        return (index, input, target)  
    else:
        return (input, target)

