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
  def __init__(self, width=12, depth=3, chroma_depth=1, bias=False):
    super(BasicQuadRGBModel, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_depth = chroma_depth

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
    for d in range(self.chroma_depth):
        if d == 0:
            in_c = 2
        else:
            in_c = self.width
        if d == self.chroma_depth - 1:
            out_c = 6
        else:
            out_c = self.width
        chroma_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
        if d != self.chroma_depth - 1:
            chroma_layers[f"relu{d}"] = nn.ReLU()

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

    chroma_cost = 0
    for d in range(self.chroma_depth):
        if d == 0:
            in_c = 2
        else:
            in_c = self.width
        if d == self.chroma_depth - 1:
            out_c = 6
        else:
            out_c = self.width

        chroma_cost += in_c * out_c * 3 * 3
        if d != self.chroma_depth - 1:
            chroma_cost += out_c
        if self.bias:
            chroma_cost += out_c

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
    green_pred = th.empty(th.Size(green_pred_shape), device=mosaic.device)
    green_pred[:,0,:,:,:] = mul[:,0:self.width//2,:,:]
    green_pred[:,1,:,:,:] = mul[:,self.width//2: ,:,:]
    green_rb = green_pred.sum(2, keepdim=False)

    green_add_shape = [mosaic_shape[0], 6, mosaic_shape[2], mosaic_shape[3]]
    green_add = th.empty(th.Size(green_add_shape), device=mosaic.device)
    green_add[:,0,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,1,:,:] = green_rb[:,1,:,:] # B
    green_add[:,2,:,:] = mosaic[:,3,:,:] # GB
    green_add[:,3,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,4,:,:] = green_rb[:,0,:,:] # R
    green_add[:,5,:,:] = mosaic[:,3,:,:] # GB

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


class BasicQuadRGBV2Model(nn.Module):
  def __init__(self, width=12, depth=3, bias=False):
    super(BasicQuadRGBV2Model, self).__init__()
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

    self.chromah_filter = nn.Conv2d(1, 1, 5, padding=2, bias=self.bias)
    self.chromav_filter = nn.Conv2d(1, 1, 5, padding=2, bias=self.bias)
    self.chromaq_filter = nn.Conv2d(1, 1, 5, padding=2, bias=self.bias)

    self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor=2)
    self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor=2)

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

    mult_sumR_cost = self.width * 2 
    cost += mult_sumR_cost
    cost /= 4

    chroma_cost = 3 * (1 * 1 * 5 * 5) 
    if self.bias:
      chroma_cost += 1

    cost += chroma_cost # chroma is on full res bayer

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
    
    nn.init.xavier_normal_(self.chromah_filter.weight)
    if self.bias:
      nn.init.zeros_(self.chromah_filter.bias)
    
    nn.init.xavier_normal_(self.chromav_filter.weight)
    if self.bias:
      nn.init.zeros_(self.chromav_filter.bias)

    nn.init.xavier_normal_(self.chromaq_filter.weight)
    if self.bias:
      nn.init.zeros_(self.chromaq_filter.bias) 

  def forward(self, mosaic):
    # 1/4 resolution features
    interps = self.filter_processor(mosaic)
    weights = self.weight_processor(mosaic)

    mul = interps * weights

    mosaic_shape = list(mosaic.shape)

    green_pred_shape = [mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]]
    green_pred = th.empty(th.Size(green_pred_shape), device=mosaic.device)
    green_pred[:,0,:,:,:] = mul[:,0:self.width//2,:,:]
    green_pred[:,1,:,:,:] = mul[:,self.width//2: ,:,:]
    green_rb = green_pred.sum(2, keepdim=False)

    full_green_quad = th.empty(th.Size([mosaic_shape[0],4,mosaic_shape[2], mosaic_shape[3]]), device=mosaic.device)
    full_green_quad[:,0,:,:] = mosaic[:,0,:,:]
    full_green_quad[:,1,:,:] = green_rb[:,0,:,:]
    full_green_quad[:,2,:,:] = green_rb[:,1,:,:]
    full_green_quad[:,3,:,:] = mosaic[:,3,:,:]

    flat_green = self.pixel_shuffle1(full_green_quad)
    flat_mosaic = self.pixel_shuffle2(mosaic)

    chroma_input = flat_mosaic - flat_green

    chromah = self.chromah_filter(chroma_input) + flat_green
    chromav = self.chromav_filter(chroma_input) + flat_green
    chromaq = self.chromaq_filter(chroma_input) + flat_green

    img_shape = [mosaic_shape[0], 3, mosaic_shape[2]*2, mosaic_shape[3]*2]
    img = th.empty(th.Size(img_shape), device=mosaic.device)

    img[:,0,0::2,0::2] = chromah[:,0,0::2,0::2] 
    img[:,0,0::2,1::2] = mosaic[:,1,:,:]
    img[:,0,1::2,0::2] = chromaq[:,0,1::2,0::2]
    img[:,0,1::2,1::2] = chromav[:,0,1::2,1::2]

    img[:,1,0::2,0::2] = mosaic[:,0,:,:]
    img[:,1,0::2,1::2] = green_rb[:,0,:,:]
    img[:,1,1::2,0::2] = green_rb[:,1,:,:]
    img[:,1,1::2,1::2] = mosaic[:,3,:,:]

    img[:,2,0::2,0::2] = chromav[:,0,0::2,0::2]
    img[:,2,0::2,1::2] = chromaq[:,0,0::2,1::2]
    img[:,2,1::2,0::2] = mosaic[:,2,:,:]
    img[:,2,1::2,1::2] = chromah[:,0,1::2,1::2]

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

class MultiresQuadRGBModel(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, bias=False, scale_factor=2):
    super(MultiresQuadRGBModel, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_depth = chroma_depth
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

    chroma_layers = OrderedDict()
    for d in range(self.chroma_depth):
        if d == 0:
            in_c = 2
        else:
            in_c = self.width
        if d == self.chroma_depth - 1:
            out_c = 6
        else:
            out_c = self.width
        chroma_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
        if d != self.chroma_depth - 1:
            chroma_layers[f"relu{d}"] = nn.ReLU()

    self.chroma_processor = nn.Sequential(chroma_layers)


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

    chroma_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 2
      else:
        in_c = self.width
      if d == self.chroma_depth - 1:
        out_c = 6
      else:
        out_c = self.width

      chroma_cost += in_c * out_c * 3 * 3
      if d != self.chroma_depth - 1:
        chroma_cost += out_c
      if self.bias:
        chroma_cost += out_c

    cost = (lowres_cost + fullres_cost + mult_sumR_cost + chroma_cost) 
    cost /= 4

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
       
    for l in self.chroma_processor:
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

    res_embed = self.width//2 # half of channels are for GatR, half for GatB

    mosaic_shape = list(mosaic.shape)

    green_pred_shape = [mosaic_shape[0], 2, self.width, mosaic_shape[2], mosaic_shape[3]]
    green_pred = th.empty(th.Size(green_pred_shape), device=mosaic.device)
    green_pred[:,0,0:res_embed,:,:] = mul[:,0:res_embed,:,:]
    green_pred[:,0,res_embed:, :,:] = mul[:,res_embed*2:res_embed*3,:,:]

    green_pred[:,1,0:res_embed,:,:] = mul[:,res_embed:res_embed*2,:,:]
    green_pred[:,1,res_embed:, :,:] = mul[:,res_embed*3:,:,:]

    green_rb = green_pred.sum(2, keepdim=False)

    green_add_shape = [mosaic_shape[0], 6, mosaic_shape[2], mosaic_shape[3]]
    green_add = th.empty(th.Size(green_add_shape), device=mosaic.device)
    green_add[:,0,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,1,:,:] = green_rb[:,1,:,:] # B
    green_add[:,2,:,:] = mosaic[:,3,:,:] # GB
    green_add[:,3,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,4,:,:] = green_rb[:,0,:,:] # R
    green_add[:,5,:,:] = mosaic[:,3,:,:] # GB

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


"""
Green model:
does a cheap sequence of 3x3 cons on fullres bayer to predict green
feeds in ratio of bayer / pred_green1 stacked with mosaic to a sequence 
of 3x3 convs on downsampled bayer quad -> produce mixing weights for
the more expensive lowres and fullres interpolations.

Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green.

chroma model: 
Weight predictor is on downsampled green predictions 
Interp is on full res r - g / b - g stacked with the green predictions 
"""
class MultiresQuadGreenRatioModel(nn.Module):
  def __init__(self, cheap_depth=1, width=12, depth=2, fixed_cheap=False, bias=False, scale_factor=2):
    super(MultiresQuadGreenRatioModel, self).__init__()
    self.cheap_depth = cheap_depth
    self.width = width
    self.depth = depth
    self.bias = bias
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor + 2 # change to scale_factor * 2 
    self.fixed_cheap = fixed_cheap

    if self.fixed_cheap:
      self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    # cheap green interp
    cheap_interp_layers = OrderedDict()
    if self.fixed_cheap:
      cheap_interp_layers["conv"] = nn.Conv2d(1, 1, 5, padding=2, bias=self.bias)
    else: 
      for d in range(self.cheap_depth):
        if d == 0:
          in_c = 4
        else:
          in_c = self.width
        if d == self.cheap_depth - 1:
          out_c = 2
        else:
          out_c = self.width

        cheap_interp_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
        if d != self.cheap_depth-1:
          cheap_interp_layers[f"relu{d}"] = nn.ReLU()

    lowres_layers = OrderedDict()
    lowres_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      lowres_layers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        lowres_layers[f"relu{d}"] = nn.ReLU()

    lowres_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') # nn.ConvTranspose2d(width, width, 2, stride=2, groups=width, bias=False)

    fullres_layers = OrderedDict()
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      fullres_layers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        fullres_layers[f"relu{d}"] = nn.ReLU()

    selector_layers = OrderedDict()
    selector_layers["downsample"] = nn.Conv2d(6, 6, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.width*2

      selector_layers[f"conv{d}"] = nn.Conv2d(in_c, width*2, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        selector_layers[f"relu{d}"] = nn.ReLU()

    selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    selector_layers["softmax"] = nn.Softmax(dim=1)

    self.cheap_filter = nn.Sequential(cheap_interp_layers)
    self.lowres_filter = nn.Sequential(lowres_layers)
    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

  def compute_cost(self):
    cheap_interp_cost = 0
    if self.fixed_cheap:
      cheap_interp_cost = (4 / 2) * 4 # 4 tap filter only at R and B locs. Multiply by 4 cus on fullres bayer
    else: 
      for d in range(self.cheap_depth):
        if d == 0:
          in_c = 4
        else:
          in_c = self.width
        if d == self.cheap_depth-1:
          out_c = 2
        else:
          out_c = self.width

        cheap_interp_cost += in_c * out_c * 3 * 3 
        if d != self.cheap_depth - 1:
          cheap_interp_cost += self.width # relu
        if self.bias:
          cheap_interp_cost += self.width # bias
    cheap_interp_cost /= 2 # only computed at R and B locations
    rb_g_ratio_cost = (DIV_COST / 2) # only computed at R and B locations

    upsample_cost = 3 
    downsample_cost = 4 * 4 * self.downsample_w * self.downsample_w

    lowres_interp_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      lowres_interp_cost += in_c * self.width * 3 * 3 
      if d != self.depth - 1:
        lowres_interp_cost += self.width
      if self.bias:
        lowres_interp_cost += self.width

    lowres_interp_cost += (upsample_cost * self.width)
    
    selector_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 6 # mosaic stacked with ratio of R,B to cheap green prediction 
      else:
        in_c = self.width*2

      selector_cost += in_c * self.width*2 * 3 * 3 
      if d != self.depth - 1:
        selector_cost += self.width*2
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

      fullres_cost += in_c * self.width * 3 * 3 
      if d != self.depth - 1:
        fullres_cost += self.width
      if self.bias:
        fullres_cost += self.width

    mult_sumR_cost = self.width * 2 # doing 2 dot products with vectors of size self.wdith (multadds count as 1) (self.width * 2) * 2
    cost = (cheap_interp_cost + rb_g_ratio_cost + lowres_cost + fullres_cost + mult_sumR_cost) 
    cost /= 4

    return cost

  def _initialize_parameters(self):
    for l in self.cheap_filter:
      print(l)
      if hasattr(l, "weight"):
        if self.fixed_cheap: # weight is 1x1x3x3 on flat bayer
          l.weight.data[...] = 0
          l.weight.data[0,0,0,1] = 0.25
          l.weight.data[0,0,1,0] = 0.25
          l.weight.data[0,0,1,1] = 1
          l.weight.data[0,0,1,2] = 0.25
          l.weight.data[0,0,2,1] = 0.25
        else:
         nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

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
    eps = 1e-10
    # 1/4 resolution features
    mosaic_shape = list(mosaic.shape)
    rb_shape = [mosaic_shape[0], 2, mosaic_shape[2], mosaic_shape[3]]

    if self.fixed_cheap: # simple averaging of just green values
      flat_mosaic = self.pixel_shuffle(mosaic)
      flat_mosaic_shape = list(flat_mosaic.shape)
      green_mosaic = th.empty(th.Size(flat_mosaic_shape), device=mosaic.device)
      green_mosaic[...] = 0
      green_mosaic[:,0,0::2,0::2] = flat_mosaic[0,0,0::2,0::2]
      green_mosaic[:,0,1::2,1::2] = flat_mosaic[0,0,1::2,1::2]

      cheap_green_pred = self.cheap_filter(green_mosaic)

      cheap_green_rb = th.empty(th.Size(rb_shape), device=mosaic.device)
      cheap_green_rb[:,0,:,:] = cheap_green_pred[:,0,0::2,1::2]
      cheap_green_rb[:,1,:,:] = cheap_green_pred[:,0,1::2,0::2]
    else: # learned interp on full bayer mosaic
      cheap_green_rb = self.cheap_filter(mosaic)

    rb = th.empty(th.Size(rb_shape), device=mosaic.device)
    rb[:,0,:,:] = mosaic[:,1,:,:] 
    rb[:,1,:,:] = mosaic[:,2,:,:] 

    rb_g_ratio = th.div(rb, cheap_green_rb + eps)

    selector_input = th.cat((rb_g_ratio, mosaic), 1)

    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(selector_input)

    interps = th.cat((lowres_interp, fullres_interp), 1)
    mul = interps * selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 
   
    green_shape = [mosaic_shape[0], 1, mosaic_shape[2]*2, mosaic_shape[3]*2]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,0::2,0::2] = mosaic[:,0,:,:]
    green[:,0,0::2,1::2] = green_rb[:,0,:,:]
    green[:,0,1::2,0::2] = green_rb[:,1,:,:]
    green[:,0,1::2,1::2] = mosaic[:,3,:,:]

    return green 



"""
Green model:
Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green.

Selector model:
first layer takes interp with N channels as input to a 3x3 grouped conv with 
1 channel in 2 channels out per group to mimic computing finite differences

second layer is 1x1 grouped conv takes 2 channels in and 1 channel out per group
to mimic max operator over finite differences 

third layer is 1x1 conv takes N channels in, N channels out to mimic min
operator over gradients of each interp direction

this produces the mixing weights for the lowres and fullres interpolations.

chroma model: 
Weight predictor is on downsampled green predictions 
Interp is on full res r - g / b - g stacked with the green predictions 
"""
class MultiresQuadGreenV2Model(nn.Module):
  def __init__(self, cheap_depth=1, width=12, depth=2, fixed_cheap=False, bias=False, scale_factor=2):
    super(MultiresQuadGreenV2Model, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor * 2  

    lowres_layers = OrderedDict()
    lowres_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=(downsample_w-scale_factor)//2, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      lowres_layers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        lowres_layers[f"relu{d}"] = nn.ReLU()

    lowres_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') # nn.ConvTranspose2d(width, width, 2, stride=2, groups=width, bias=False)

    fullres_layers = OrderedDict()
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      fullres_layers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        fullres_layers[f"relu{d}"] = nn.ReLU()

    selector_layers = OrderedDict()
    N = self.width*2
    D = 2
    selector_layers["conv1"] = nn.Conv2d(N, N*D, 3, groups=N, padding=1, bias=self.bias)
    selector_layers["relu1"] = nn.ReLU()
    selector_layers["conv2"] = nn.Conv2d(N*D, N, 1, groups=N, padding=0, bias=self.bias)
    selector_layers["relu2"] = nn.ReLU()
    selector_layers["conv3"] = nn.Conv2d(N, N, 1, padding=0, bias=self.bias)
    selector_layers["softmax"] = nn.Softmax(dim=1)

    self.lowres_filter = nn.Sequential(lowres_layers)
    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

  def compute_cost(self):
    upsample_cost = 3 
    downsample_cost = 4 * 4 * self.downsample_w * self.downsample_w

    lowres_interp_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      lowres_interp_cost += in_c * self.width * 3 * 3 
      if d != self.depth - 1:
        lowres_interp_cost += self.width
      if self.bias:
        lowres_interp_cost += self.width

    lowres_interp_cost += (upsample_cost * self.width)
    lowres_cost = lowres_interp_cost / (self.scale_factor*self.scale_factor)

    selector_cost = 0
    D = 2
    selector_c1_cost = self.width*2 * D * 3 * 3 
    if self.bias:
      selector_c1_cost += self.width*2 * D
    selector_r1_cost = self.width*2 * D

    selector_c2_cost = D * self.width*2 * 1 * 1 
    if self.bias:
      selector_c2_cost += self.width*2
    selector_r2_cost = self.width*2 

    selector_c3_cost = self.width*2 * self.width*2 * 1 * 1

    softmax_cost = self.width*2 * (LOGEXP_COST + DIV_COST + ADD_COST)
    
    selector_cost = sum([selector_c1_cost, selector_r1_cost, selector_c2_cost, selector_r2_cost, selector_c3_cost])
    selector_cost += softmax_cost

    fullres_cost = 0
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      fullres_cost += in_c * self.width * 3 * 3 
      if d != self.depth - 1:
        fullres_cost += self.width
      if self.bias:
        fullres_cost += self.width

    mult_sumR_cost = self.width * 2 # doing 2 dot products with vectors of size self.wdith (multadds count as 1) (self.width * 2) * 2
    cost = (lowres_cost + fullres_cost + selector_cost + mult_sumR_cost) 
    cost /= 4

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
    eps = 1e-10
    # 1/4 resolution features
    mosaic_shape = list(mosaic.shape)
    rb_shape = [mosaic_shape[0], 2, mosaic_shape[2], mosaic_shape[3]]
    rb = th.empty(th.Size(rb_shape), device=mosaic.device)
    rb[:,0,:,:] = mosaic[:,1,:,:] 
    rb[:,1,:,:] = mosaic[:,2,:,:] 

    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    interps = th.cat((lowres_interp, fullres_interp), 1)

    #input to selector is stacked lowres and full res interps
    selectors = self.selector(interps)
    
    mul = interps * selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 
   
    green_shape = [mosaic_shape[0], 1, mosaic_shape[2]*2, mosaic_shape[3]*2]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,0::2,0::2] = mosaic[:,0,:,:]
    green[:,0,0::2,1::2] = green_rb[:,0,:,:]
    green[:,0,1::2,0::2] = green_rb[:,1,:,:]
    green[:,0,1::2,1::2] = mosaic[:,3,:,:]

    return green 


"""
Green model:
Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green.

Selector model:
first layer takes lowres interp with N channels as input to a 3x3 grouped conv with 
1 channel in 2 channels out per group to mimic computing finite differences

second layer is 1x1 grouped conv takes 2 channels in and 1 channel out per group
to mimic max operator over finite differences 

third layer is 1x1 conv takes N channels in, N channels out to mimic min
operator over gradients of each interp direction

finally we upsample the result and tile it twice.
this produces the mixing weights for the lowres and fullres interpolations.

"""
class MultiresQuadGreenV3Model(nn.Module):
  def __init__(self, width=12, depth=2, bias=False, scale_factor=2):
    super(MultiresQuadGreenV3Model, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor + 2 # change to scale_factor * 2 

    lowres_layers = OrderedDict()
    lowres_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=(downsample_w-scale_factor)//2, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      lowres_layers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        lowres_layers[f"relu{d}"] = nn.ReLU()

    self.lowres_upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear') # nn.ConvTranspose2d(width, width, 2, stride=2, groups=width, bias=False)

    fullres_layers = OrderedDict()
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      fullres_layers[f"conv{d}"] = nn.Conv2d(in_c, width, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        fullres_layers[f"relu{d}"] = nn.ReLU()

    selector_layers = OrderedDict()
    N = self.width # takes lowres interps as input 
    D = 2
    selector_layers["conv1"] = nn.Conv2d(N, N*D, 3, groups=N, padding=1, bias=self.bias)
    selector_layers["relu1"] = nn.ReLU()
    selector_layers["conv2"] = nn.Conv2d(N*D, N, 1, groups=N, padding=0, bias=self.bias)
    selector_layers["relu2"] = nn.ReLU()
    selector_layers["conv3"] = nn.Conv2d(N, N, 1, padding=0, bias=self.bias)
    selector_layers["softmax"] = nn.Softmax(dim=1)
    selector_layers["upsampler"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

    self.lowres_filter = nn.Sequential(lowres_layers)
    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

  def compute_cost(self):
    upsample_cost = 3 
    downsample_cost = 4 * 4 * self.downsample_w * self.downsample_w

    lowres_interp_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      lowres_interp_cost += in_c * self.width * 3 * 3 
      if d != self.depth - 1:
        lowres_interp_cost += self.width
      if self.bias:
        lowres_interp_cost += self.width

    lowres_interp_cost += (upsample_cost * self.width)
    lowres_cost = lowres_interp_cost / (self.scale_factor*self.scale_factor)

    selector_cost = 0
    D = 2
    selector_c1_cost = self.width * D * 3 * 3 
    if self.bias:
      selector_c1_cost += self.width * D
    selector_r1_cost = self.width * D

    selector_c2_cost = D * self.width * 1 * 1 
    if self.bias:
      selector_c2_cost += self.width
    selector_r2_cost = self.width 

    selector_c3_cost = self.width * self.width * 1 * 1

    softmax_cost = self.width * (LOGEXP_COST + DIV_COST + ADD_COST)
    
    selector_cost = sum([selector_c1_cost, selector_r1_cost, selector_c2_cost, selector_r2_cost, selector_c3_cost])
    selector_cost += softmax_cost
    selector_cost /= (self.scale_factor*self.scale_factor)

    fullres_cost = 0
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width

      fullres_cost += in_c * self.width * 3 * 3 
      if d != self.depth - 1:
        fullres_cost += self.width
      if self.bias:
        fullres_cost += self.width

    mult_sumR_cost = self.width * 2 # doing 2 dot products with vectors of size self.wdith (multadds count as 1) (self.width * 2) * 2
    cost = (lowres_cost + fullres_cost + selector_cost + mult_sumR_cost) 
    cost /= 4

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
    eps = 1e-10
    # 1/4 resolution features
    mosaic_shape = list(mosaic.shape)

    lowres_interp = self.lowres_filter(mosaic) 
    lowres_upsampled_interp = self.lowres_upsampler(lowres_interp)
    fullres_interp = self.fullres_filter(mosaic)
    interps = th.cat((lowres_upsampled_interp, fullres_interp), 1)

    # input to selector is lowres interps
    selectors = self.selector(lowres_interp)
    tiled_selectors = th.repeat_interleave(selectors, repeats=2, dim=0).reshape(interps.shape)
    
    mul = interps * tiled_selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 
   
    green_shape = [mosaic_shape[0], 1, mosaic_shape[2]*2, mosaic_shape[3]*2]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,0::2,0::2] = mosaic[:,0,:,:]
    green[:,0,0::2,1::2] = green_rb[:,0,:,:]
    green[:,0,1::2,0::2] = green_rb[:,1,:,:]
    green[:,0,1::2,1::2] = mosaic[:,3,:,:]

    return green 


