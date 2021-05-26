import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data
from imageio import imread
from config import IMG_H, IMG_W
from dataset_util import ids_from_file, get_cropped_img_size
from mosaic_gen import bayer
from superres_mosaic_gen import lowres_bayer, bicubic_downsample
import math


class SGreenQuadDataset(data.Dataset):
  def __init__(self, data_file=None, return_images=False, return_index=False):
    
    if data_file:
        self.list_IDs = ids_from_file(data_file)
    else: 
        self.input_IDs = ids_from_file(input_data_file) # patch filenames
        self.target_IDs = ids_from_file(target_data_file)

    self.return_images = return_images
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
    
    # print(f"downsampled_size size after cropping : {lowres_img.shape}")
    # print(f"first valid input loc: {first_valid_in_loc} last valid input loc {last_valid_in_loc}")
    # print(f"first valid output loc: {first_valid_out_loc} last valid output loc {last_valid_out_loc}")

    lowres_mosaic = bayer(lowres_img)
    lowres_mosaic = torch.sum(lowres_mosaic, 0, keepdim=True)

    quad_size = list(lowres_mosaic.shape)
    quad_size[0] = 4
    quad_size[1] //= 2
    quad_size[2] //= 2

    bayer_quad = np.zeros(quad_size)
    bayer_quad[0,:,:] = lowres_mosaic[0,0::2,0::2]
    bayer_quad[1,:,:] = lowres_mosaic[0,0::2,1::2]
    bayer_quad[2,:,:] = lowres_mosaic[0,1::2,0::2]
    bayer_quad[3,:,:] = lowres_mosaic[0,1::2,1::2]

    bayer_quad = torch.Tensor(bayer_quad)
    input = bayer_quad
  
    target = np.expand_dims(img[1,...], axis=0)
    # crop out the valid region of the fullres image 
    target = target[:,first_valid_in_loc:last_valid_in_loc+1, first_valid_in_loc:last_valid_in_loc+1]
    
    # print(f"output target shape: {target.shape}")

    target = torch.Tensor(target)

    if self.return_images:
        cropped_fullres = img[:,first_valid_in_loc:last_valid_in_loc+1, first_valid_in_loc:last_valid_in_loc+1]
        if self.return_index:
            return (index, input, cropped_fullres, lowres_img, target)
        else:
            return (input, cropped_fullres, lowres_img, target)
    elif self.return_index:
        return (index, input, target)  
    else:
        return (input, target)


class SRGBQuadDataset(data.Dataset):
  def __init__(self, data_file=None, return_images=False, return_index=False):
    
    if data_file:
        self.list_IDs = ids_from_file(data_file)
    else: 
        self.input_IDs = ids_from_file(input_data_file) # patch filenames
        self.target_IDs = ids_from_file(target_data_file)

    self.return_images = return_images
    self.return_index = return_index

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
    # print("-----")
    image_f = self.list_IDs[index]
    img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)

    if len(img.shape) == 2: #grayscale image
        print(f"grayscale {img.shape}")
        img = np.expand_dims(img, 2)
        img = np.tile(img, (1,1,3))

    img = np.transpose(img, [2, 0, 1])

    h = img.shape[-2]
    w = img.shape[-1]
    # crop image so that the downsampled image h, w will be a multiple of 12
    # print(f"input shape {img.shape}")
    new_h = get_cropped_img_size(h)
    new_w = get_cropped_img_size(w)

    hcrop = h - new_h
    wcrop = w - new_w

    img = img[:, hcrop:, wcrop:]

    # print(f"cropped shape {img.shape}")

    (hlowres_start, hlowres_end), (hfullres_start, hfullres_end), pad = self.get_start_end_locs(img.shape[-2])
    (wlowres_start, wlowres_end), (wfullres_start, wfullres_end), pad = self.get_start_end_locs(img.shape[-1])

    # add a batch dimension to the image to use torch convolutions for downsampling
    batched_img = torch.Tensor(img).unsqueeze(0)
    lowres_img = bicubic_downsample(batched_img, factor=2, pad=pad)[0]
    # print(f"downsampled size {lowres_img.shape}")
    # print(f"using pad {pad} for kernel radius {kw} and scale factor {SCALE}")
    # print(f"computed downsampled size: {downsampled_size}  downsampled_size size before cropping : {lowres_img.shape}")
    
    # crop out the valid region of the lowres image
    # print(f"h start lowres loc: {hlowres_start} h end lowres loc: {hlowres_end}")

    lowres_img = lowres_img[:,hlowres_start:hlowres_end+1, wlowres_start:wlowres_end+1]
    # print(f"lowres shape {lowres_img.shape}")
    # print(f"downsampled_size size after cropping : {lowres_img.shape}")
    # print(f"first valid input loc: {first_valid_in_loc} last valid input loc {last_valid_in_loc}")
    # print(f"first valid output loc: {first_valid_out_loc} last valid output loc {last_valid_out_loc}")

    lowres_mosaic = bayer(lowres_img)
    lowres_mosaic = torch.sum(lowres_mosaic, 0, keepdim=True)

    quad_size = list(lowres_mosaic.shape)
    quad_size[0] = 4
    quad_size[1] //= 2
    quad_size[2] //= 2

    quad_h = quad_size[1]
    quad_w = quad_size[2]

    bayer_quad = np.zeros(quad_size)
    bayer_quad[0,:,:] = lowres_mosaic[0,0::2,0::2]
    bayer_quad[1,:,:] = lowres_mosaic[0,0::2,1::2]
    bayer_quad[2,:,:] = lowres_mosaic[0,1::2,0::2]
    bayer_quad[3,:,:] = lowres_mosaic[0,1::2,1::2]

    redblue_bayer = np.zeros((2, quad_h, quad_w))
    green_grgb = np.zeros((2, quad_h, quad_w))

    redblue_bayer[0,:,:] = bayer_quad[1,:,:]
    redblue_bayer[1,:,:] = bayer_quad[2,:,:]

    target = img
    # crop out the valid region of the fullres image 
    target = target[:,hfullres_start:hfullres_end+1, wfullres_start:wfullres_end+1]

    # print(f"target shape {target.shape}")

    bayer_quad = torch.Tensor(bayer_quad)
    redblue_bayer = torch.Tensor(redblue_bayer)
    target = torch.Tensor(target)

    input = (bayer_quad, redblue_bayer)

    if self.return_index:
        return (index, input, target)  
    else:
        return (input, target)



class SBaselineRGBQuadDataset(data.Dataset):
  def __init__(self, data_file=None, return_index=False, crop=True):
    
    if data_file:
        self.list_IDs = ids_from_file(data_file)
    else: 
        self.input_IDs = ids_from_file(input_data_file) # patch filenames
        self.target_IDs = ids_from_file(target_data_file)

    self.return_index = return_index
    self.crop = crop

  def __len__(self):
    return len(self.list_IDs)

  def get_crop(self, w):
    for i in range(w, 0, -1):
        if i % 12 == 0:
            return w-i

  def __getitem__(self, index):
    lr_img_f = self.list_IDs[index]
    hr_img_f = lr_img_f.rstrip("LR.png") + "HR.png"
    print(hr_img_f)

    lr_img = np.array(imread(lr_img_f)).astype(np.float32) / (2**8-1)
    hr_img = np.array(imread(hr_img_f)).astype(np.float32) / (2**8-1)

    if len(hr_img.shape) == 2: #grayscale image
        print(f"grayscale {hr_img.shape}")
        hr_img = np.expand_dims(hr_img, 2)
        hr_img = np.tile(hr_img, (1,1,3))
        lr_img = np.expand_dims(lr_img, 2)
        lr_img = np.tile(lr_img, (1,1,3))

    if self.crop:
        h_crop = self.get_crop(lr_img.shape[0])
        w_crop = self.get_crop(lr_img.shape[1])
        if h_crop > 0:
            print(f"cropped: h {lr_img.shape[1]} crop: {h_crop} ")
            lr_img = lr_img[h_crop:, :, :]
            hr_img = hr_img[h_crop*2:, :, :]
        if w_crop > 0:
            lr_img = lr_img[:, w_crop:, :]
            hr_img = hr_img[:, w_crop*2:, :]

    hr_img = np.transpose(hr_img, [2, 0, 1])
    lr_img = np.transpose(lr_img, [2, 0, 1])
 
    lr_img = torch.Tensor(lr_img)
    hr_img = torch.Tensor(hr_img)

    lowres_mosaic = bayer(lr_img)
    lowres_mosaic = torch.sum(lowres_mosaic, 0, keepdim=True)

    quad_size = list(lowres_mosaic.shape)
    quad_size[0] = 4
    quad_size[1] //= 2
    quad_size[2] //= 2

    quad_h = quad_size[1]
    quad_w = quad_size[2]

    bayer_quad = np.zeros(quad_size)
    bayer_quad[0,:,:] = lowres_mosaic[0,0::2,0::2]
    bayer_quad[1,:,:] = lowres_mosaic[0,0::2,1::2]
    bayer_quad[2,:,:] = lowres_mosaic[0,1::2,0::2]
    bayer_quad[3,:,:] = lowres_mosaic[0,1::2,1::2]

    redblue_bayer = np.zeros((2, quad_h, quad_w))
    green_grgb = np.zeros((2, quad_h, quad_w))

    redblue_bayer[0,:,:] = bayer_quad[1,:,:]
    redblue_bayer[1,:,:] = bayer_quad[2,:,:]

    bayer_quad = torch.Tensor(bayer_quad)
    redblue_bayer = torch.Tensor(redblue_bayer)
    target = torch.Tensor(hr_img)

    input = (bayer_quad, redblue_bayer)

    if self.return_index:
        return (index, input, target)  
    else:
        return (input, target)



if __name__ == "__main__":
    import argparse
    from dataset import FastDataLoader
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=str)
    args = parser.parse_args()

    dataset = GreenQuadDataset(data_file=args.filelist, return_index=True, return_images=True)
    n = len(dataset)
    indices = list(range(n))
    image_names = ids_from_file(args.filelist)

    dataloader = FastDataLoader(
        dataset, batch_size=1,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
        pin_memory=True, num_workers=1)

    for i, (index, input, cropped_fullres, lowres_img, target) in enumerate(dataloader):
        image_f = image_names[index]
        image_id = image_f.split("/")[-1].strip(".png")
        pil_lowres_img = np.transpose(lowres_img[0].numpy()*255, [1,2,0]).astype(np.uint8)
        pil_fullres_img = np.transpose(cropped_fullres[0].numpy()*255, [1,2,0]).astype(np.uint8)

        lowres_f = image_id+"-lowres.png" 
        fullres_f = image_id+"-fullres.png"

        Image.fromarray(pil_lowres_img).save(lowres_f)
        Image.fromarray(pil_fullres_img).save(fullres_f)

        if i > 5:
            break




