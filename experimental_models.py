import torch as th
import torch.nn as nn
from collections import OrderedDict
import sys 

from cost import ADD_COST, MUL_COST, RELU_COST, LOGEXP_COST, DIV_COST

IMG_H = 128
IMG_W = 128



class BasicQuadGreenModel(nn.Module):
  def __init__(self, width=12, depth=3, bias=False):
    super(BasicQuadGreenModel, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    dlayers = OrderedDict()

    for d in range(self.depth):
        if d == 0:
            in_c = 4
        else:
            in_c = self.width
        dlayers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
        dlayers[f"relu{d}"] = nn.ReLU()
    dlayers[f"softmax"] = nn.Softmax(dim=1)

    self.weight_processor = nn.Sequential(dlayers)

    flayers = OrderedDict()

    for d in range(self.depth):
        if d == 0:
            in_c = 4
        else:
            in_c = self.width
        flayers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
        flayers[f"relu{d}"] = nn.ReLU()

    self.filter_processor = nn.Sequential(flayers)

  def compute_cost(self):
    cost = 0
    for d in range(self.depth):
        if d == 0:
            in_c = 4
        else:
            in_c = self.width

        cost += in_c * self.width * 3 * 3 + self.width
        if self.bias:
            cost += self.width

    cost *= 2 # interp and weight trunks
    cost += (self.width * (LOGEXP_COST + DIV_COST + ADD_COST))

    mult_sumR_cost = self.width * 2 
    cost += mult_sumR_cost
    cost /= 4
    return cost
    
  def _initialize_parameters(self):
    for l in self.filter_processor:
      print(l)
      if hasattr(l, "weight"):
        nn.init.xavier_normal_(l.weight)
        if self.bias:
            nn.init.zeros_(l.bias)
    for l in self.weight_processor:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
            nn.init.zeros_(l.bias)
        
  def forward(self, mosaic):
    # 1/4 resolution features
    interps = self.filter_processor(mosaic)
    weights = self.weight_processor(mosaic)

    mul = interps * weights

    out_shape = list(mosaic.shape)
    out_shape[1] = self.width // 2
    out_shape[2] *= 2
    out_shape[3] *= 2

    green = th.empty(th.Size(out_shape), device=mosaic.device)
    green[:,:,0::2,1::2] = mul[:,0:self.width//2,:,:]
    green[:,:,1::2,0::2] = mul[:,self.width//2: ,:,:]
    flat_green = green.sum(1, keepdim=True)
    flat_green[:,0,0::2,0::2] = mosaic[:,0,:,:]
    flat_green[:,0,1::2,1::2] = mosaic[:,3,:,:]

    return flat_green 

class BasicQuadRGBModel(nn.Module):
  def __init__(self, width=12, depth=3, bias=False):
    super(BasicQuadGreenModel, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    dlayers = OrderedDict()

    for d in range(self.depth):
        if d == 0:
            in_c = 4
        else:
            in_c = self.width
        dlayers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
        dlayers[f"relu{d}"] = nn.ReLU()
    dlayers[f"softmax"] = nn.Softmax(dim=1)

    self.weight_processor = nn.Sequential(dlayers)

    flayers = OrderedDict()

    for d in range(self.depth):
        if d == 0:
            in_c = 4
        else:
            in_c = self.width
        flayers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
        flayers[f"relu{d}"] = nn.ReLU()

    self.filter_processor = nn.Sequential(flayers)

    chroma_layers = OrderedDict()
    chroma_layers["conv"] = nn.Conv2d(2, 6, 3, padding=1, bias=self.bias)
    chroma_layers["relu"] = nn.ReLU()
    self.chroma_processor = nn.Sequential(chroma_layers)

  def compute_cost(self):
    cost = 0
    for d in range(self.depth):
        if d == 0:
            in_c = 4
        else:
            in_c = self.width

        cost += in_c * self.width * 3 * 3 + self.width
        if self.bias:
            cost += self.width

    cost *= 2 # green interp and chroma interp and weight trunks
    cost += (self.width * (LOGEXP_COST + DIV_COST + ADD_COST)) # softmax cost

    chroma_cost = 2 * 6 * 3 * 3 + 6
    if self.bias:
      chroma_cost += 6

    cost += chroma_cost

    mult_sumR_cost = self.width * 2 
    cost += mult_sumR_cost
    cost /= 4
    return cost
    
  def _initialize_parameters(self):
    for l in self.filter_processor:
      print(l)
      if hasattr(l, "weight"):
        nn.init.xavier_normal_(l.weight)
        if self.bias:
            nn.init.zeros_(l.bias)
    for l in self.weight_processor:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
            nn.init.zeros_(l.bias)
    for l in self.chroma_processor:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
            nn.init.zeros_(l.bias)
        
  def forward(self, mosaic):
    # 1/4 resolution features
    interps = self.filter_processor(mosaic)
    weights = self.weight_processor(mosaic)

    mul = interps * weights

    mosaic_shape = list(mosaic.shape)

    green_pred_shape = [mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]]
    green_rb = th.empty(th.Size(green_pred_shape), device=mosaic.device)
    green_rb[:,0,:,:,:] = mul[:,0:self.width//2,:,:]
    green_rb[:,1,:,:,:] = mul[:,self.width//2: ,:,:]
    green_rb = green.sum(2, keepdim=False)

    green_add = [mosaic_shape[0], 6, mosaic_shape[2], mosaic_shape[3]]
    green_add = th.empty(th.Size(full_green_shape), device=mosaic.device)
    green_add[:,0,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,1,:,:] = green_rb[:,1,:,:] # B
    green_add[:,2,:,:] = mosaic[:,3,:,:] # GB
    green_add[:,3,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,4,:,:] = green_rb[:,0,:,:] # R
    green_add[:,5,:,:] = mosaic[:,3,:,:] # R

    rb_shape = green_rb.shape 
    rb = th.empty(th.Size(rb_shape), device=mosaic.device)
    rb[:,0,:,:] = mosaic[:,1,:,:] 
    rb[:,1,:,:] = mosaic[:,2,:,:] 
    chroma_input = rb - green_rb

    chroma_diff = self.chroma_processor(chroma_input)
    chroma_pred = chroma_diff + green_add

    img_shape = [mosaic_shape[0], 3, mosaic_shape[2]*2, mosaic_shape[3]*2]
    img = th.empty(th.Size(img_shape), device=mosaic.device)

    img[:,0,0::2,0::2] = chroma_pred[:,0,:,:]
    img[:,0,0::2,1::2] = mosaic[:,1,:,:]
    img[:,0,1::2,0::2] = chroma_pred[:,1,:,:]
    img[:,0,1::2,1::2] = chroma_pred[:,2,:,:]

    img[:,1,0::2,0::2] = mosaic[:,0,:,:]
    img[:,1,0::2,1::2] = green_rb[:,0,:,:]
    img[:,1,1::2,0::2] = green_rb[:,1,:,:]
    img[:,1,1::2,1::2] = mosaic[:,3,:,:]

    img[:,2,0::2,0::2] = chroma_pred[:,3,:,:]
    img[:,2,0::2,1::2] = chroma_pred[:,4,:,:]
    img[:,2,1::2,0::2] = mosaic[:,2,:,:]
    img[:,2,1::2,1::2] = chroma_pred[:,5,:,:]

    return img 


class MultiresQuadGreenModel(nn.Module):
  def __init__(self, width=12, depth=2, bias=False, scale_factor=2):
    super(MultiresQuadGreenModel, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor + 2

    lowres_layers = OrderedDict()
    lowres_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      lowres_layers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
      lowres_layers[f"relu{d}"] = nn.ReLU()

    lowres_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') # nn.ConvTranspose2d(width, width, 2, stride=2, groups=width, bias=False)

    fullres_layers = OrderedDict()
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      fullres_layers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
      fullres_layers[f"relu{d}"] = nn.ReLU()

    selector_layers = OrderedDict()
    selector_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width*2

      selector_layers[f"conv{d}"] = nn.Conv2d(in_c, width*2, 3, padding=1, bias=self.bias)
      selector_layers[f"relu{d}"] = nn.ReLU()

    selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    selector_layers["softmax"] = nn.Softmax(dim=1)

    self.lowres_filter = nn.Sequential(lowres_layers)
    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

  def compute_cost(self):
    upsample_cost = 3 # WHAT SHOULD THIS BE FOR A FACTOR OF 4?
    downsample_cost = 4 * 4 * self.downsample_w * self.downsample_w

    lowres_interp_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      lowres_interp_cost += in_c * self.width * 3 * 3 + self.width
      if self.bias:
        lowres_interp_cost += self.width

    lowres_interp_cost += (upsample_cost * self.width)
    
    selector_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width*2

      selector_cost += in_c * self.width*2 * 3 * 3  + self.width*2
      if self.bias:
        selector_cost += self.width*2

    selector_cost += (upsample_cost * self.width*2)

    softmax_cost = self.width*2 * (LOGEXP_COST + DIV_COST + ADD_COST)

    lowres_cost = (lowres_interp_cost + selector_cost + softmax_cost) / (self.scale_factor*self.scale_factor)

    fullres_cost = 0
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      fullres_cost += in_c * self.width * 3 * 3 + self.width
      if self.bias:
        fullres_cost += self.width

    mult_sumR_cost = (self.width * 2) * 2
    cost = (lowres_cost + fullres_cost + mult_sumR_cost) / 4
    return cost

  def _initialize_parameters(self):
    for l in self.lowres_filter:
      print(l)
      if hasattr(l, "weight"):
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.fullres_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.selector:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)
       
  def forward(self, mosaic):
    # 1/4 resolution features
    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(mosaic)

    interps = th.cat((lowres_interp, fullres_interp), 1)
    mul = interps * selectors

    out_shape = list(mosaic.shape)
    out_shape[1] = self.width 
    out_shape[2] *= 2
    out_shape[3] *= 2

    green = th.empty(th.Size(out_shape), device=mosaic.device)

    res_embed = self.width//2 # half of channels are for GatR, half for GatB

    green[:,0:res_embed,0::2,1::2] = mul[:,0:res_embed,:,:]
    green[:,res_embed: ,0::2,1::2] = mul[:,res_embed*2:res_embed*3,:,:]

    green[:,0:res_embed,1::2,0::2] = mul[:,res_embed:res_embed*2,:,:]
    green[:,res_embed: ,1::2,0::2] = mul[:,res_embed*3:,:,:]

    flat_green = green.sum(1, keepdim=True)
    flat_green[:,0,0::2,0::2] = mosaic[:,0,:,:]
    flat_green[:,0,1::2,1::2] = mosaic[:,3,:,:]

    return flat_green 


class LowResSelectorFullResInterpGreenModel(nn.Module):
  def __init__(self, depth=2, width=12, bias=False):
    super(LowResSelectorFullResInterpGreenModel, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias

    fullres_layers = OrderedDict()
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      fullres_layers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
      fullres_layers[f"relu{d}"] = nn.ReLU()

    selector_layers = OrderedDict()
    selector_layers["downsample"] = nn.Conv2d(4, 4, 4, stride=2, padding=1, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      selector_layers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
      selector_layers[f"relu{d}"] = nn.ReLU()

    selector_layers["upsample"] = nn.Upsample(scale_factor=2, mode='bilinear') 
    selector_layers["softmax"] = nn.Softmax(dim=1)

    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

  def _initialize_parameters(self):
    for l in self.fullres_filter:
      print(l)
      if hasattr(l, "weight"):
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)
   
    for l in self.selector:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)
        
  def compute_cost(self):
    upsample_cost = 3 # +6 * MUL_COST + 6 * DIV_COST + 10 * MUL_COST + DIV_COST
    downsample_cost = 4 * 4 * 4 * 4
    
    selector_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width*2

      selector_cost += in_c * self.width * 3 * 3  + self.width
      if self.bias:
        selector_cost += self.width

    selector_cost += (upsample_cost * self.width)
    softmax_cost = self.width * (LOGEXP_COST + DIV_COST + ADD_COST)
    lowres_cost = (selector_cost + softmax_cost) / 4

    fullres_cost = 0
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      fullres_cost += in_c * self.width * 3 * 3 + self.width
      if self.bias:
        fullres_cost += self.width

    mult_sumR_cost = (self.width) * 2
    cost = (lowres_cost + fullres_cost + mult_sumR_cost) / 4
    return cost

  def forward(self, mosaic):
    # 1/4 resolution features
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(mosaic)

    mul = fullres_interp * selectors

    out_shape = list(mosaic.shape)
    out_shape[1] = self.width // 2
    out_shape[2] *= 2
    out_shape[3] *= 2

    green = th.empty(th.Size(out_shape), device=mosaic.device)

    res_embed = self.width//2 # half of channels are for GatR, half for GatB

    green[:,:,0::2,1::2] = mul[:,0:res_embed,:,:]
    green[:,:,1::2,0::2] = mul[:,res_embed:, :,:]

    flat_green = green.sum(1, keepdim=True)
    flat_green[:,0,0::2,0::2] = mosaic[:,0,:,:]
    flat_green[:,0,1::2,1::2] = mosaic[:,3,:,:]

    return flat_green 




