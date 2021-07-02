import torch 
import torch.nn as nn
from collections import OrderedDict
import sys 
import numpy as np
from cost import ADD_COST, MUL_COST, RELU_COST, LOGEXP_COST, DIV_COST

IMG_H = 128
IMG_W = 128

def compute_cost(k, filters):
    green_weight_conv_cost = k**2 * filters
    green_interp_conv_cost = green_weight_conv_cost
    stable_cost = ADD_COST * (filters-1) + ADD_COST * filters # max and sub
    sofmax_cost = filters * (LOGEXP_COST + DIV_COST + ADD_COST)
    green_cost = green_weight_conv_cost + green_interp_conv_cost + stable_cost + sofmax_cost + (MUL_COST* (filters-1)) 

    sub_green_cost = 3
    add_green_cost = 3
    chroma_filter_cost = 3 * k**2  
    chroma_cost = sub_green_cost + chroma_filter_cost + add_green_cost

    cost = green_cost + chroma_cost
    print(f"gweight {green_weight_conv_cost} gfilter {green_interp_conv_cost} stable_cost {stable_cost} sofmax_cost {sofmax_cost} chromafilter {chroma_filter_cost}")
    return cost


class Demosaicer(nn.Module):
  def __init__(self, k, filters):
    super(Demosaicer, self).__init__()
    self.DirectionModel = nn.Conv2d(1, filters, k, bias=False, padding=k//2)
    self.FilterModel = nn.Conv2d(1, filters, k, bias=False, padding=k//2)
    self.normalizer = nn.Softmax(dim=1)

  def _initialize_parameters(self):
    torch.nn.init.xavier_normal_(self.DirectionModel.weight)
    torch.nn.init.xavier_normal_(self.FilterModel.weight)

  def forward(self, flatbayer):
    direction_weights = self.DirectionModel(flatbayer)
    # subtract max for stability
    max_weights, _ = torch.max(direction_weights, dim=1, keepdim=True)
    stable_weights = direction_weights - max_weights
    normed_weights = self.normalizer(stable_weights)
  
    filter_outputs = self.FilterModel(flatbayer)

    weighted_outputs = normed_weights * filter_outputs
    out = torch.sum(weighted_outputs, dim=1)
    out = out.unsqueeze(1)

    return out


class GradientHalide(nn.Module):
  def __init__(self, k, filters, pretrained=True):
    super(GradientHalide, self).__init__()

    self.Green = Demosaicer(k, filters)

    self.ChromaV = nn.Conv2d(1, 1, k, bias=False, padding=k//2)
    self.ChromaH = nn.Conv2d(1, 1, k, bias=False, padding=k//2)
    self.ChromaQ = nn.Conv2d(1, 1, k, bias=False, padding=k//2)
   
    # define masks
    self.chromaHBmask = torch.zeros((IMG_H, IMG_W))
    self.chromaHBmask[1::2,1::2] = 1

    self.chromaVRmask = self.chromaHBmask

    self.chromaHRmask = torch.zeros((IMG_H, IMG_W))
    self.chromaHRmask[0::2,0::2] = 1

    self.chromaVBmask = self.chromaHRmask

    self.chromaQRmask = torch.zeros((IMG_H, IMG_W))
    self.chromaQRmask[1::2,0::2] = 1
    self.chromaQBmask = torch.zeros((IMG_H, IMG_W))
    self.chromaQBmask[0::2,1::2] = 1

  def _initialize_parameters(self):
    self.Green._initialize_parameters()
    torch.nn.init.xavier_normal_(self.ChromaV.weight)
    torch.nn.init.xavier_normal_(self.ChromaH.weight)
    torch.nn.init.xavier_normal_(self.ChromaQ.weight)

  def to_gpu(self, gpu_id):
    self.chromaHBmask = self.chromaHBmask.to(device=f"cuda:{gpu_id}")
    self.chromaHRmask = self.chromaHRmask.to(device=f"cuda:{gpu_id}")
    self.chromaQRmask = self.chromaQRmask.to(device=f"cuda:{gpu_id}")
    self.chromaQBmask = self.chromaQBmask.to(device=f"cuda:{gpu_id}")
    self.chromaVRmask = self.chromaVRmask.to(device=f"cuda:{gpu_id}")
    self.chromaVBmask = self.chromaVBmask.to(device=f"cuda:{gpu_id}")
    self.Gmask = (self.chromaQRmask + self.chromaQBmask).to(device=f"cuda:{gpu_id}")


  def forward(self, bayers):
    flatbayer, threechan_bayer = bayers

    img_h = flatbayer.shape[-2]
    img_w = flatbayer.shape[-1]
    # define masks
    chromaHBmask = torch.zeros((img_h, img_w))
    chromaHBmask[1::2,1::2] = 1

    chromaVRmask = chromaHBmask

    chromaHRmask = torch.zeros((img_h, img_w))
    chromaHRmask[0::2,0::2] = 1

    chromaVBmask = chromaHRmask

    chromaQRmask = torch.zeros((img_h, img_w))
    chromaQRmask[1::2,0::2] = 1
    chromaQBmask = torch.zeros((img_h, img_w))
    chromaQBmask[0::2,1::2] = 1

    Gmask = (chromaQRmask + chromaQBmask)

    green = self.Green(flatbayer)
    green = green * Gmask + threechan_bayer[:,1,...].unsqueeze(1)

    chroma_input = flatbayer - green
    chromav = self.ChromaV(chroma_input) + green
    chromah = self.ChromaH(chroma_input) + green
    chromaq = self.ChromaQ(chroma_input) + green

    red = chromav * chromaVRmask \
          + chromah * chromaHRmask \
          + chromaq * chromaQRmask \
          + threechan_bayer[:,0,...].unsqueeze(1)

    blue = chromav * chromaVBmask \
          + chromah * chromaHBmask \
          + chromaq * chromaQBmask \
          + threechan_bayer[:,2,...].unsqueeze(1)

    image = torch.cat([red, green, blue], dim=1)
  
    return image
