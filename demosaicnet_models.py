import torch as th
import torch.nn as nn
from collections import OrderedDict
import sys 
import numpy as np

from cost import ADD_COST, MUL_COST, RELU_COST

IMG_H = 128
IMG_W = 128

"""
slightly modified from mgharbi 
https://github.com/mgharbi/demosaicnet/blob/master/demosaicnet/modules.py
"""
class FullDemosaicknet(nn.Module):

  def __init__(self, depth=15, width=64, pretrained=True, pad=True): 
    super(FullDemosaicknet, self).__init__()

    self.depth = depth
    self.width = width

    if pad:
      pad = 1
    else:
      pad = 0

    layers = OrderedDict([
        ("pack_mosaic", nn.Conv2d(3, 4, 2, stride=2)),  # Downsample 2x2 to re-establish translation invariance
      ])
    for i in range(depth):
      n_out = width
      n_in = width
      if i == 0:
        n_in = 4
      """ for version with residual masking """
      if i == depth-1:
        n_out = 2*width

      layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3, padding=pad)
      layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)

    self.main_processor = nn.Sequential(layers)
    self.residual_predictor = nn.Conv2d(width, 12, 1)
    self.upsampler = nn.ConvTranspose2d(12, 3, 2, stride=2, groups=3)

    self.fullres_processor = nn.Sequential(OrderedDict([
      ("post_conv", nn.Conv2d(6, width, 3, padding=pad)),
      ("post_relu", nn.ReLU(inplace=True)),
      ("output", nn.Conv2d(width, 3, 1)),
      ]))

    self.mask = th.ones((3, IMG_H, IMG_W))
    self.mask[0,0::2,1::2] = 0

    self.mask[1,0::2,0::2] = 0
    self.mask[1,1::2,1::2] = 0

    self.mask[2,1::2,0::2] = 0


  def _initialize_parameters(self):
    for l in self.main_processor:
      print(l)
      if hasattr(l, "weight"):
        nn.init.xavier_normal_(l.weight)
    nn.init.xavier_normal_(self.residual_predictor.weight)
    nn.init.xavier_normal_(self.upsampler.weight)

    for l in self.fullres_processor:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        
  def to_gpu(self, gpu_id):
    self.mask = self.mask.to(device=f"cuda:{gpu_id}")

  def forward(self, mosaic):
    """Demosaicks a Bayer image.
    Args:
      mosaic (th.Tensor):  input Bayer mosaic
    Returns:
      th.Tensor: the demosaicked image
    """

    # 1/4 resolution features
    """ my version without the residuual asking """
    # features = self.main_processor(mosaic)
    # residual = self.residual_predictor(features)

    # # Match mosaic and residual
    # upsampled = self.upsampler(residual)
    # #cropped = _crop_like(mosaic, upsampled)

    # packed = th.cat([mosaic, upsampled], 1)  # skip connection
    # output = self.fullres_processor(packed)

    # rgb = output * self.mask + mosaic

    # return rgb


    features = self.main_processor(mosaic)
    filters, masks = features[:, 0:self.width], features[:, self.width:2*self.width]
    filtered = filters * masks
    residual = self.residual_predictor(filtered)

    # Match mosaic and residual
    upsampled = self.upsampler(residual)
    cropped = _crop_like(mosaic, upsampled)

    packed = th.cat([cropped, upsampled], 1)  # skip connection
    output = self.fullres_processor(packed)

    rgb = output * self.mask + mosaic
    return rgb



  def compute_cost(self):
    cost = bayer_packer_cost()
    cost += main_processor_cost(self.depth, self.width)
    cost += residual_pred_cost(self.width)
    cost += upsampler_cost()

    cost /= 4 # lowres 
    final_out_c = 3
    cost += fullres_processor_cost(self.width, final_out_c)
    return cost 


"""
slightly modified from mgharbi 
https://github.com/mgharbi/demosaicnet/blob/master/demosaicnet/modules.py
"""
class GreenDemosaicknet(nn.Module):

  def __init__(self, depth=15, width=64, pretrained=True, pad=True):
    super(GreenDemosaicknet, self).__init__()

    self.depth = depth
    self.width = width

    if pad:
      pad = 1
    else:
      pad = 0

    layers = OrderedDict([
        ("pack_mosaic", nn.Conv2d(3, 4, 2, stride=2, bias=False)),  # Downsample 2x2 to re-establish translation invariance
      ])
    for i in range(depth):
      n_out = width
      n_in = width
      if i == 0:
        n_in = 4
      layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3, padding=pad, bias=False)
      layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)

    self.main_processor = nn.Sequential(layers)
    self.residual_predictor = nn.Conv2d(width, 12, 1, bias=False)
    self.upsampler = nn.ConvTranspose2d(12, 3, 2, stride=2, groups=3, bias=False)

    self.fullres_processor = nn.Sequential(OrderedDict([
      ("post_conv", nn.Conv2d(6, width, 3, padding=pad, bias=False)),
      ("post_relu", nn.ReLU(inplace=True)),
      ("output", nn.Conv2d(width, 1, 1, bias=False)),
      ]))

    self.mask = th.zeros((IMG_H, IMG_W))
    self.mask[0::2,1::2] = 1
    self.mask[1::2,0::2] = 1

  def _initialize_parameters(self):
    for l in self.main_processor:
      print(l)
      if hasattr(l, "weight"):
        nn.init.xavier_normal_(l.weight)
    nn.init.xavier_normal_(self.residual_predictor.weight)
    nn.init.xavier_normal_(self.upsampler.weight)

    for l in self.fullres_processor:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)

  def to_gpu(self, gpu_id):
    self.mask = self.mask.to(device=f"cuda:{gpu_id}")

  def forward(self, mosaic):
    # 1/4 resolution features
    features = self.main_processor(mosaic)
    residual = self.residual_predictor(features)

    # Match mosaic and residual
    upsampled = self.upsampler(residual)
    #cropped = _crop_like(mosaic, upsampled)

    packed = th.cat([mosaic, upsampled], 1)  # skip connection
    output = self.fullres_processor(packed)

    bayer = th.sum(mosaic, dim=1, keepdim=True)
    #green = output * self.mask + (self.chroma_mask*bayer[:,0,...]).unsqueeze(1)
    green = output * self.mask + bayer[:,0,...].unsqueeze(1)

    return green

  def compute_cost(self):
    cost = bayer_packer_cost()
    cost += main_processor_cost(self.depth, self.width)
    cost += residual_pred_cost(self.width)
    cost += upsampler_cost()

    cost /= 4 # lowres 
    final_out_c = 1
    cost += fullres_processor_cost(self.width, final_out_c)
    return cost 

def bayer_packer_cost():
  in_c = 3
  out_c = 4
  cost = in_c * out_c * 2 * 2 * MUL_COST + out_c
  return cost

def main_processor_cost(layers, width):
  k = 3
  in_c = 4
  cost = in_c * width * k * k * MUL_COST + width
  for l in range(layers-1):
    cost += width * width * k * k * MUL_COST + width
    cost += width * RELU_COST
  return cost

def residual_pred_cost(width):
  k = 1
  in_c = width
  out_c = 12
  cost = in_c * out_c * k * k * MUL_COST + out_c
  return cost 

def upsampler_cost():
  in_c = 12
  out_c = 3
  k = 2
  groups = 3

  cost = groups * (in_c // groups) * (out_c // groups) * k * k * MUL_COST + out_c
  return cost

def fullres_processor_cost(width, final_out_c):
  in_c = 6
  out_c = width
  k = 3
  cost = in_c * out_c * k * k * MUL_COST + out_c
  cost += out_c * RELU_COST

  in_c = width 
  out_c = final_out_c
  k = 1
  cost += (in_c * out_c * k * k * MUL_COST + out_c)/2 # only need to compute final conv at two locations for green
  return cost 


def _crop_like(src, tgt):
    """Crop a source image to match the spatial dimensions of a target.
    Args:
        src (th.Tensor or np.ndarray): image to be cropped
        tgt (th.Tensor or np.ndarray): reference image
    """
    src_sz = np.array(src.shape)
    tgt_sz = np.array(tgt.shape)

    # Assumes the spatial dimensions are the last two
    crop = (src_sz[-2:]-tgt_sz[-2:])
    crop_t = crop[0] // 2
    crop_b = crop[0] - crop_t
    crop_l = crop[1] // 2
    crop_r = crop[1] - crop_l
    crop //= 2

    #print(f"crop_t {crop_t} crop_b {crop_b} crop_l {crop_l} crop_r {crop_r}")

    if (np.array([crop_t, crop_b, crop_r, crop_l])> 0).any():
        return src[..., crop_t:src_sz[-2]-crop_b, crop_l:src_sz[-1]-crop_r]
    else:
        return src