import torch as th
import torch.nn as nn
from collections import OrderedDict
import sys 

from cost import ADD_COST, MUL_COST, RELU_COST, LOGEXP_COST, DIV_COST

IMG_H = 128
IMG_W = 128


"""
Uses sequence of 3x3 convs on bayer quad to predict chroma
"""
class BasicQuadRGBModel(nn.Module):
  def __init__(self, width=12, depth=3, chroma_depth=1, chroma_width=12, bias=False):
    super(BasicQuadRGBModel, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_depth = chroma_depth
    self.chroma_width = chroma_width

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
            in_c = self.chroma_width
        if d == self.chroma_depth - 1:
            out_c = 6
        else:
            out_c = self.chroma_width
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

    cost *= 2 # green interp and weight trunks cost the same
    cost += (self.width * (LOGEXP_COST + DIV_COST + ADD_COST)) # softmax cost
    mult_sumR_cost = self.width * 2 
    cost += mult_sumR_cost

    chroma_cost = 0
    for d in range(self.chroma_depth):
        if d == 0:
            in_c = 2
        else:
            in_c = self.width
        if d == self.chroma_depth - 1:
            out_c = 6
        else:
            out_c = self.chroma_width

        chroma_cost += in_c * out_c * 3 * 3
        if d != self.chroma_depth - 1:
            chroma_cost += out_c # no relu after final conv
        if self.bias:
            chroma_cost += out_c

    cost += chroma_cost

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

    mul_reshaped = th.reshape(mul, mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])
    green_rb = mul_reshaped.sum(2, keepdim=False)

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


"""
Uses 5x5 filter on flat bayer to predict ChromaH, ChromaV, ChromaQ
"""
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

    cost *= 2 # green interp and weight trunks cost the same
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

    mul_reshaped = th.reshape(mul, mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])
    green_rb = mul_reshaped.sum(2, keepdim=False)

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


"""
Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green.
A third sequence of 3x3 convs on downsampled bayer quad is used to produce mixing weights for
the lowres and fullres interpolations.

Uses sequence of 3x3 convs on bayer quad to predict Chroma
"""
class MultiresQuadRGBModel(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=16, bias=False, scale_factor=2):
    super(MultiresQuadRGBModel, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_depth = chroma_depth
    self.chroma_width = chroma_width
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor + 2 # change to scale_factor * 2 

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
    selector_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width*2

      selector_layers[f"conv{d}"] = nn.Conv2d(in_c, width*2, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
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
            in_c = self.chroma_width
        if d == self.chroma_depth - 1:
            out_c = 6
        else:
            out_c = self.chroma_width
        chroma_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
        if d != self.chroma_depth - 1:
            chroma_layers[f"relu{d}"] = nn.ReLU()

    self.chroma_processor = nn.Sequential(chroma_layers)

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
    
    selector_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
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

    chroma_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 2
      else:
        in_c = self.chroma_width
      if d == self.chroma_depth - 1:
        out_c = 6
      else:
        out_c = self.chroma_width

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
    mosaic_shape = list(mosaic.shape)

    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(mosaic)

    interps = th.cat((lowres_interp, fullres_interp), 1)
    mul = interps * selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 

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

"""
Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green. 
A third sequence of 3x3 convs on downsampled bayer quad is used to produce mixing weights for
the lowres and fullres interpolations.
 
Repeats this same structure for predicting chroma
"""
class MultiresQuadRGBV2Model(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=12, bias=False, scale_factor=2):
    super(MultiresQuadRGBV2Model, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_width = chroma_width
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

    chroma_lowres_layers = OrderedDict()
    chroma_lowres_layers["downsample"] = nn.Conv2d(2, 2, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 2
      else:
        in_c = self.chroma_width

      chroma_lowres_layers[f"conv{d}"] = nn.Conv2d(in_c, self.chroma_width, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth - 1:
        chroma_lowres_layers[f"relu{d}"] = nn.ReLU()

    chroma_lowres_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') # nn.ConvTranspose2d(width, width, 2, stride=2, groups=width, bias=False)

    chroma_fullres_layers = OrderedDict()
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 2
      else:
        in_c = self.chroma_width

      chroma_fullres_layers[f"conv{d}"] = nn.Conv2d(in_c, self.chroma_width, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth - 1:
        chroma_fullres_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers = OrderedDict()
    chroma_selector_layers["downsample"] = nn.Conv2d(2, 2, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 2
      else:
        in_c = self.chroma_width*2

      chroma_selector_layers[f"conv{d}"] = nn.Conv2d(in_c, chroma_width*2, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth - 1:
        chroma_selector_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    chroma_selector_layers["softmax"] = nn.Softmax(dim=1)

    self.chroma_lowres_filter = nn.Sequential(chroma_lowres_layers)
    self.chroma_fullres_filter = nn.Sequential(chroma_fullres_layers)
    self.chroma_selector = nn.Sequential(chroma_selector_layers)


  def multires_module_cost(scale_factor, depth, width, bias=False, final_relu=True):
    downsample_w = scale_factor + 2

    upsample_cost = 3 
    downsample_cost = 4 * 4 * downsample_w * downsample_w

    lowres_interp_cost = downsample_cost
    for d in range(depth):
      if d == 0:
        in_c = 4
      else:
        in_c = width

      lowres_interp_cost += in_c * width * 3 * 3 
      if final_relu or d < depth-1:
        lowres_interp_cost += width
      if bias:
        lowres_interp_cost += width

    lowres_interp_cost += (upsample_cost * width)
    
    selector_cost = downsample_cost
    for d in range(depth):
      if d == 0:
        in_c = 4
      else:
        in_c = width*2

      selector_cost += in_c * width*2 * 3 * 3  
      if final_relu or d < depth-1:
        selector_cost += width*2
      if bias:
        selector_cost += width*2

    selector_cost += (upsample_cost * width*2)
    softmax_cost = width*2 * (LOGEXP_COST + DIV_COST + ADD_COST)
    lowres_cost = (lowres_interp_cost + selector_cost + softmax_cost) / (scale_factor*scale_factor)

    fullres_cost = 0
    for d in range(depth):
      if d == 0:
        in_c = 4
      else:
        in_c = width

      fullres_cost += in_c * width * 3 * 3 
      if final_relu or d < depth-1:
        fullres_cost += width
      if bias:
        fullres_cost += width

    mult_sumR_cost = (width * 2) 

    cost = (lowres_cost + fullres_cost + mult_sumR_cost) / 4
    return cost

  def compute_cost(self):
    green_cost = MultiresQuadRGBV2Model.multires_module_cost(self.scale_factor, self.depth, self.width, self.bias, final_relu=True)
    chroma_cost = MultiresQuadRGBV2Model.multires_module_cost(self.scale_factor, self.chroma_depth, self.chroma_width, self.bias, final_relu=False)
    cost = chroma_cost + green_cost
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
       
    for l in self.chroma_lowres_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)
   
    for l in self.chroma_fullres_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.chroma_selector:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

  def forward(self, mosaic):
    mosaic_shape = list(mosaic.shape)

    # 1/4 resolution features
    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(mosaic)

    interps = th.cat((lowres_interp, fullres_interp), 1)
    mul = interps * selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 

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

    chroma_lowres_interp = self.chroma_lowres_filter(chroma_input)
    chroma_fullres_interp = self.chroma_fullres_filter(chroma_input)
    chroma_selectors = self.chroma_selector(chroma_input)
    
    chroma_interps = th.cat((chroma_lowres_interp, chroma_fullres_interp), 1)
    chroma_mul = chroma_interps * chroma_selectors

    chroma_lowres_mul, chroma_fullres_mul = th.split(chroma_mul, self.chroma_width, dim=1) # n x 2*cw x w x h --> 2 tensors of n x cw x w x h size
    chroma_lmul_reshaped = th.reshape(chroma_lowres_mul, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3]))  # n x 6 x cw/6, x w x h
    chroma_fmul_reshaped = th.reshape(chroma_fullres_mul, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3])) # n x 6 x cw/6, x w x h

    chroma_weighted_interps = th.cat((chroma_lmul_reshaped, chroma_fmul_reshaped), 2)  # n x 6 x 2*cw/6, x w x h
    chroma_diff = chroma_weighted_interps.sum(2, keepdim=False) # n x 6 x w x h

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


"""
Same as V2 model except Chroma prediction does not predict r - g or b - g 
instead predicts red and blue values directly
"""
class MultiresQuadRGBV3Model(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=12, bias=False, scale_factor=2):
    super(MultiresQuadRGBV3Model, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_width = chroma_width
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
    selector_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width*2

      selector_layers[f"conv{d}"] = nn.Conv2d(in_c, width*2, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        selector_layers[f"relu{d}"] = nn.ReLU()

    selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    selector_layers["softmax"] = nn.Softmax(dim=1)

    self.lowres_filter = nn.Sequential(lowres_layers)
    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

    chroma_lowres_layers = OrderedDict()
    chroma_lowres_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.chroma_width

      chroma_lowres_layers[f"conv{d}"] = nn.Conv2d(in_c, self.chroma_width, 3, padding=1, bias=self.bias)    
      if d != self.chroma_depth-1:
        chroma_lowres_layers[f"relu{d}"] = nn.ReLU()
    chroma_lowres_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') # nn.ConvTranspose2d(width, width, 2, stride=2, groups=width, bias=False)

    chroma_fullres_layers = OrderedDict()
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.chroma_width

      chroma_fullres_layers[f"conv{d}"] = nn.Conv2d(in_c, self.chroma_width, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth-1:
        chroma_fullres_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers = OrderedDict()
    chroma_selector_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.chroma_width*2

      chroma_selector_layers[f"conv{d}"] = nn.Conv2d(in_c, chroma_width*2, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth - 1:
        chroma_selector_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    chroma_selector_layers["softmax"] = nn.Softmax(dim=1)

    self.chroma_lowres_filter = nn.Sequential(chroma_lowres_layers)
    self.chroma_fullres_filter = nn.Sequential(chroma_fullres_layers)
    self.chroma_selector = nn.Sequential(chroma_selector_layers)


  def multires_module_cost(scale_factor, depth, width, first_in_c, bias=False, final_relu=True):
    downsample_w = scale_factor + 2

    upsample_cost = 3 
    downsample_cost = first_in_c * first_in_c * downsample_w * downsample_w

    lowres_interp_cost = downsample_cost
    for d in range(depth):
      if d == 0:
        in_c = first_in_c
      else:
        in_c = width

      lowres_interp_cost += in_c * width * 3 * 3 
      if final_relu or d < depth-1:
        lowres_interp_cost += width
      if bias:
        lowres_interp_cost += width

    lowres_interp_cost += (upsample_cost * width)
    
    selector_cost = downsample_cost
    for d in range(depth):
      if d == 0:
        in_c = first_in_c
      else:
        in_c = width*2

      selector_cost += in_c * width*2 * 3 * 3  
      if d < depth-1:
        selector_cost += width*2
      if bias:
        selector_cost += width*2

    selector_cost += (upsample_cost * width*2)
    softmax_cost = width*2 * (LOGEXP_COST + DIV_COST + ADD_COST)
    lowres_cost = (lowres_interp_cost + selector_cost + softmax_cost) / (scale_factor*scale_factor)

    fullres_cost = 0
    for d in range(depth):
      if d == 0:
        in_c = first_in_c
      else:
        in_c = width

      fullres_cost += in_c * width * 3 * 3 
      if final_relu or d < depth-1:
        fullres_cost += width
      if bias:
        fullres_cost += width

    mult_sumR_cost = (width * 2) 

    cost = (lowres_cost + fullres_cost + mult_sumR_cost) / 4
    return cost

  def compute_cost(self):
    green_cost = MultiresQuadRGBV3Model.multires_module_cost(self.scale_factor, self.depth, self.width, 4, self.bias, final_relu=False)
    chroma_cost = MultiresQuadRGBV3Model.multires_module_cost(self.scale_factor, self.chroma_depth, self.chroma_width, 2, self.bias, final_relu=False)
    cost = chroma_cost + green_cost
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
       
    for l in self.chroma_lowres_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)
   
    for l in self.chroma_fullres_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.chroma_selector:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

  def forward(self, mosaic):
    mosaic_shape = list(mosaic.shape)

    # 1/4 resolution features
    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(mosaic)

    interps = th.cat((lowres_interp, fullres_interp), 1)
    mul = interps * selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 

    chroma_lowres_interp = self.chroma_lowres_filter(mosaic)
    chroma_fullres_interp = self.chroma_fullres_filter(mosaic)
    chroma_selectors = self.chroma_selector(mosaic)
    
    chroma_interps = th.cat((chroma_lowres_interp, chroma_fullres_interp), 1)
    chroma_mul = chroma_interps * chroma_selectors

    chroma_lowres_mul, chroma_fullres_mul = th.split(chroma_mul, self.chroma_width, dim=1) # n x 2*cw x w x h --> 2 tensors of n x cw x w x h size
    chroma_lmul_reshaped = th.reshape(chroma_lowres_mul, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3]))  # n x 6 x cw/6, x w x h
    chroma_fmul_reshaped = th.reshape(chroma_fullres_mul, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3])) # n x 6 x cw/6, x w x h

    chroma_weighted_interps = th.cat((chroma_lmul_reshaped, chroma_fmul_reshaped), 2)  # n x 6 x 2*cw/6, x w x h
    chroma_pred = chroma_weighted_interps.sum(2, keepdim=False) # n x 6 x w x h

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



"""
Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green.
A third sequence of 3x3 convs on downsampled bayer quad is used to produce mixing weights for
the lowres and fullres interpolations.

Uses sequence of 3x3 convs on r - g / b - g stacked with the green predictions to predict Chroma
"""
class MultiresQuadRGBV4Model(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=16, bias=False, scale_factor=2):
    super(MultiresQuadRGBV4Model, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_depth = chroma_depth
    self.chroma_width = chroma_width
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor + 2 # change to scale_factor * 2 

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
    selector_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width*2

      selector_layers[f"conv{d}"] = nn.Conv2d(in_c, width*2, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        selector_layers[f"relu{d}"] = nn.ReLU()

    selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    selector_layers["softmax"] = nn.Softmax(dim=1)

    self.lowres_filter = nn.Sequential(lowres_layers)
    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

    chroma_layers = OrderedDict()
    for d in range(self.chroma_depth):
        if d == 0:
            in_c = 6
        else:
            in_c = self.chroma_width
        if d == self.chroma_depth - 1:
            out_c = 6
        else:
            out_c = self.chroma_width
        chroma_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
        if d != self.chroma_depth - 1:
            chroma_layers[f"relu{d}"] = nn.ReLU()

    self.chroma_processor = nn.Sequential(chroma_layers)

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
    
    selector_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
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

    chroma_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width
      if d == self.chroma_depth - 1:
        out_c = 6
      else:
        out_c = self.chroma_width

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
    mosaic_shape = list(mosaic.shape)

    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(mosaic)

    interps = th.cat((lowres_interp, fullres_interp), 1)
    mul = interps * selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 

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
    rb_min_g = rb - green_rb

    green_shape = [mosaic_shape[0], 4, mosaic_shape[2], mosaic_shape[3]]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,:,:] = mosaic[:,0,:,:]
    green[:,1,:,:] = green_rb[:,0,:,:]
    green[:,2,:,:] = green_rb[:,1,:,:]
    green[:,3,:,:] = mosaic[:,3,:,:]

    chroma_input = th.cat((rb_min_g, green), 1)

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



"""
Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green.
A third sequence of 3x3 convs on downsampled bayer quad is used to produce mixing weights for
the lowres and fullres interpolations.

Uses basic model on r - g / b - g stacked with the green predictions to predict Chroma
"""
class MultiresQuadRGBV5Model(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=16, bias=False, scale_factor=2):
    super(MultiresQuadRGBV5Model, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_depth = chroma_depth
    self.chroma_width = chroma_width
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor + 2 # change to scale_factor * 2 

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
    selector_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width*2

      selector_layers[f"conv{d}"] = nn.Conv2d(in_c, width*2, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        selector_layers[f"relu{d}"] = nn.ReLU()

    selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    selector_layers["softmax"] = nn.Softmax(dim=1)

    self.lowres_filter = nn.Sequential(lowres_layers)
    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

    chroma_interp_layers = OrderedDict()
    for d in range(self.chroma_depth):
        if d == 0:
            in_c = 6
        else:
            in_c = self.chroma_width

        out_c = self.chroma_width
        chroma_interp_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
        if d != self.chroma_depth - 1:
            chroma_interp_layers[f"relu{d}"] = nn.ReLU()

    chroma_weight_layers = OrderedDict()
    for d in range(self.chroma_depth):
        if d == 0:
            in_c = 6
        else:
            in_c = self.chroma_width

        out_c = self.chroma_width
        chroma_weight_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
        if d != self.chroma_depth - 1:
            chroma_weight_layers[f"relu{d}"] = nn.ReLU()
    chroma_weight_layers["softmax"] = nn.Softmax(dim=1)
    
    self.chroma_filter = nn.Sequential(chroma_interp_layers)
    self.chroma_selector = nn.Sequential(chroma_weight_layers)

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
    
    selector_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
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

    chroma_selector_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width

      chroma_selector_cost += in_c * self.chroma_width * 3 * 3 
      if d != self.depth - 1:
        chroma_selector_cost += self.chroma_width # relu
      if self.bias:
        chroma_selector_cost += self.chroma_width

    softmax_cost = self.chroma_width * (LOGEXP_COST + DIV_COST + ADD_COST)
    chroma_selector_cost += softmax_cost

    chroma_interp_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width
      
      chroma_interp_cost += in_c * self.chroma_width * 3 * 3
      if d != self.chroma_depth - 1:
        chroma_interp_cost += self.chroma_width
      if self.bias:
        chroma_interp_cost += self.chroma_width

    chroma_mult_sumR_cost = self.chroma_width * 2
    chroma_cost = chroma_selector_cost + chroma_interp_cost + chroma_mult_sumR_cost
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
       
    for l in self.chroma_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.chroma_selector:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

  def forward(self, mosaic):
    # 1/4 resolution features
    mosaic_shape = list(mosaic.shape)

    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(mosaic)

    interps = th.cat((lowres_interp, fullres_interp), 1)
    mul = interps * selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 

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
    rb_min_g = rb - green_rb

    green_shape = [mosaic_shape[0], 4, mosaic_shape[2], mosaic_shape[3]]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,:,:] = mosaic[:,0,:,:]
    green[:,1,:,:] = green_rb[:,0,:,:]
    green[:,2,:,:] = green_rb[:,1,:,:]
    green[:,3,:,:] = mosaic[:,3,:,:]

    chroma_input = th.cat((rb_min_g, green), 1)

    chroma_interps = self.chroma_filter(chroma_input)
    chroma_selector = self.chroma_selector(chroma_input)

    chroma_weighted = chroma_interps * chroma_selector
    chroma_weighted_reshaped = th.reshape(chroma_weighted, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3]))
    chroma_diff = chroma_weighted_reshaped.sum(2, keepdim=False)

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


"""
Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green.
A third sequence of 3x3 convs on downsampled bayer quad is used to produce mixing weights for
the lowres and fullres interpolations.

chroma model: 
Weight predictor is on downsampled r - g / b - g stacked with the green predictions 
Interp is on full res r - g / b - g stacked with the green predictions 
"""
class MultiresQuadRGBV6Model(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=16, bias=False, scale_factor=2):
    super(MultiresQuadRGBV6Model, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_depth = chroma_depth
    self.chroma_width = chroma_width
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor + 2 # change to scale_factor * 2 

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
    selector_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width*2

      selector_layers[f"conv{d}"] = nn.Conv2d(in_c, width*2, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        selector_layers[f"relu{d}"] = nn.ReLU()

    selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    selector_layers["softmax"] = nn.Softmax(dim=1)

    self.lowres_filter = nn.Sequential(lowres_layers)
    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

    chroma_interp_layers = OrderedDict()
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width

      out_c = self.chroma_width
      chroma_interp_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth - 1:
        chroma_interp_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers = OrderedDict()
    chroma_selector_layers["downsample"] = nn.Conv2d(6, 6, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width

      chroma_selector_layers[f"conv{d}"] = nn.Conv2d(in_c, self.chroma_width, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        chroma_selector_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    chroma_selector_layers["softmax"] = nn.Softmax(dim=1)

    self.chroma_filter = nn.Sequential(chroma_interp_layers)
    self.chroma_selector = nn.Sequential(chroma_selector_layers)

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
    
    selector_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
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

    chroma_downsample_cost = 6 * 6 * self.downsample_w * self.downsample_w
    chroma_selector_cost = chroma_downsample_cost
    print(f"chroma selector cost {chroma_selector_cost}")

    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width

      chroma_selector_cost += in_c * self.chroma_width * 3 * 3 
      if d != self.chroma_depth - 1:
        chroma_selector_cost += self.chroma_width # relu
      if self.bias:
        chroma_selector_cost += self.chroma_width
      print(f"chroma selector cost {chroma_selector_cost}")

    softmax_cost = self.chroma_width * (LOGEXP_COST + DIV_COST + ADD_COST)
    chroma_selector_cost += softmax_cost
    chroma_selector_cost += (upsample_cost * self.chroma_width)
    chroma_selector_cost /= (self.scale_factor*self.scale_factor)

    chroma_interp_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width
      
      chroma_interp_cost += in_c * self.chroma_width * 3 * 3
      if d != self.chroma_depth - 1:
        chroma_interp_cost += self.chroma_width
      if self.bias:
        chroma_interp_cost += self.chroma_width

    chroma_mult_sumR_cost = self.chroma_width * 2
    chroma_cost = chroma_selector_cost + chroma_interp_cost + chroma_mult_sumR_cost
    cost = (lowres_cost + fullres_cost + mult_sumR_cost + chroma_cost) 
    print(f"lowres cost {lowres_cost} fullres cost {fullres_cost} multsum r cost {mult_sumR_cost} chroma_cost {chroma_cost}")

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
       
    for l in self.chroma_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.chroma_selector:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

  def forward(self, mosaic):
    # 1/4 resolution features
    mosaic_shape = list(mosaic.shape)

    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(mosaic)

    interps = th.cat((lowres_interp, fullres_interp), 1)
    mul = interps * selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 

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
    rb_min_g = rb - green_rb

    green_shape = [mosaic_shape[0], 4, mosaic_shape[2], mosaic_shape[3]]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,:,:] = mosaic[:,0,:,:]
    green[:,1,:,:] = green_rb[:,0,:,:]
    green[:,2,:,:] = green_rb[:,1,:,:]
    green[:,3,:,:] = mosaic[:,3,:,:]

    chroma_input = th.cat((rb_min_g, green), 1)

    chroma_interps = self.chroma_filter(chroma_input)
    chroma_selector = self.chroma_selector(chroma_input)

    chroma_weighted = chroma_interps * chroma_selector
    chroma_weighted_reshaped = th.reshape(chroma_weighted, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3]))
    chroma_diff = chroma_weighted_reshaped.sum(2, keepdim=False)

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


"""
Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green.
A third sequence of 3x3 convs on downsampled bayer quad is used to produce mixing weights for
the lowres and fullres interpolations.

chroma model: 
Weight predictor is on downsampled green predictions 
Interp is on full res r - g / b - g stacked with the green predictions 
"""
class MultiresQuadRGBV7Model(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=16, bias=False, scale_factor=2):
    super(MultiresQuadRGBV7Model, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_depth = chroma_depth
    self.chroma_width = chroma_width
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor + 2 # change to scale_factor * 2 

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
    selector_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width*2

      selector_layers[f"conv{d}"] = nn.Conv2d(in_c, width*2, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        selector_layers[f"relu{d}"] = nn.ReLU()

    selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    selector_layers["softmax"] = nn.Softmax(dim=1)

    self.lowres_filter = nn.Sequential(lowres_layers)
    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

    chroma_interp_layers = OrderedDict()
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width

      out_c = self.chroma_width
      chroma_interp_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth - 1:
        chroma_interp_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers = OrderedDict()
    chroma_selector_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.chroma_width

      chroma_selector_layers[f"conv{d}"] = nn.Conv2d(in_c, self.chroma_width, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        chroma_selector_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    chroma_selector_layers["softmax"] = nn.Softmax(dim=1)

    self.chroma_filter = nn.Sequential(chroma_interp_layers)
    self.chroma_selector = nn.Sequential(chroma_selector_layers)

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
    
    selector_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
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

    chroma_downsample_cost = 4 * 4 * self.downsample_w * self.downsample_w
    chroma_selector_cost = chroma_downsample_cost
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.chroma_width

      chroma_selector_cost += in_c * self.chroma_width * 3 * 3 
      if d != self.chroma_depth - 1:
        chroma_selector_cost += self.chroma_width # relu
      if self.bias:
        chroma_selector_cost += self.chroma_width
    
      print(f"chroma selector cost {chroma_selector_cost}")

    softmax_cost = self.chroma_width * (LOGEXP_COST + DIV_COST + ADD_COST)
    chroma_selector_cost += softmax_cost
    chroma_selector_cost += (upsample_cost * self.chroma_width)
    chroma_selector_cost /= (self.scale_factor*self.scale_factor)

    chroma_interp_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width
      
      chroma_interp_cost += in_c * self.chroma_width * 3 * 3
      if d != self.chroma_depth - 1:
        chroma_interp_cost += self.chroma_width
      if self.bias:
        chroma_interp_cost += self.chroma_width

    chroma_mult_sumR_cost = self.chroma_width * 2
    chroma_cost = chroma_selector_cost + chroma_interp_cost + chroma_mult_sumR_cost
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
       
    for l in self.chroma_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.chroma_selector:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

  def forward(self, mosaic):
    # 1/4 resolution features
    mosaic_shape = list(mosaic.shape)

    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(mosaic)

    interps = th.cat((lowres_interp, fullres_interp), 1)
    mul = interps * selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 

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
    rb_min_g = rb - green_rb

    green_shape = [mosaic_shape[0], 4, mosaic_shape[2], mosaic_shape[3]]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,:,:] = mosaic[:,0,:,:]
    green[:,1,:,:] = green_rb[:,0,:,:]
    green[:,2,:,:] = green_rb[:,1,:,:]
    green[:,3,:,:] = mosaic[:,3,:,:]

    chroma_input = th.cat((rb_min_g, green), 1)

    chroma_interps = self.chroma_filter(chroma_input)
    chroma_selector = self.chroma_selector(green)

    chroma_weighted = chroma_interps * chroma_selector
    chroma_weighted_reshaped = th.reshape(chroma_weighted, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3]))
    chroma_diff = chroma_weighted_reshaped.sum(2, keepdim=False)

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


"""
Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green.
A third sequence of 3x3 convs on downsampled bayer quad is used to produce mixing weights for
the lowres and fullres interpolations.

chroma model: 
Weight predictor is on lowres r and b stacked with green predictions, learns a change of basis
in the first conv layer, then does subsequent series of convs 
Interp is on full res r - g / b - g stacked with the green predictions 
"""
class MultiresQuadRGBV8Model(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=16, bias=False, scale_factor=2):
    super(MultiresQuadRGBV8Model, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_depth = chroma_depth
    self.chroma_width = chroma_width
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor * 2 # change to scale_factor * 2 

    lowres_layers = OrderedDict()
    lowres_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=(self.downsample_w-scale_factor)//2, bias=self.bias)
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
    selector_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=(self.downsample_w-scale_factor)//2, bias=self.bias)
    for d in range(self.depth):
      if d == 0:
        in_c = 4
      else:
        in_c = self.width*2

      selector_layers[f"conv{d}"] = nn.Conv2d(in_c, width*2, 3, padding=1, bias=self.bias)
      if d != self.depth-1:
        selector_layers[f"relu{d}"] = nn.ReLU()

    selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    selector_layers["softmax"] = nn.Softmax(dim=1)

    self.lowres_filter = nn.Sequential(lowres_layers)
    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

    chroma_interp_layers = OrderedDict()
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width

      out_c = self.chroma_width
      chroma_interp_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth - 1:
        chroma_interp_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers = OrderedDict()
    chroma_selector_layers["downsample"] = nn.Conv2d(6, 6, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.chroma_depth+1):
      if d == 0:
        in_c = 6
        out_c = 6 # try learning change of basis
        k = 1
      elif d == 1:
        in_c = 6
        out_c = self.chroma_width
        k = 3
      else:
        in_c = self.chroma_width
        out_c = self.chroma_width
        k = 3

      chroma_selector_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, k, padding=k//2, bias=self.bias)
      if d != self.depth-1:
        chroma_selector_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    chroma_selector_layers["softmax"] = nn.Softmax(dim=1)

    self.chroma_filter = nn.Sequential(chroma_interp_layers)
    self.chroma_selector = nn.Sequential(chroma_selector_layers)

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
    
    selector_cost = downsample_cost
    for d in range(self.depth):
      if d == 0:
        in_c = 4
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

    chroma_downsample_cost = 6 * 6 * self.downsample_w * self.downsample_w
    chroma_selector_cost = chroma_downsample_cost
    print(f"chroma selector cost {chroma_selector_cost}")

    for d in range(self.chroma_depth): # +1):
      # if d == 0:
      #   in_c = 6 # try to learn a change of basis
      #   out_c = 6
      #   k = 1
      # elif d == 1:
      #   in_c = 6
      #   out_c = self.chroma_width
      #   k = 3
      # else:
      #   in_c = self.chroma_width
      #   out_c = self.chroma_width
      #   k = 3
      if d == 0:
        in_c = 6
        out_c = self.chroma_width
      else:
        in_c = self.chroma_width
        out_c = self.chroma_width
      k = 3

      chroma_selector_cost += in_c * out_c * k * k
      print(f"chroma selector cost {chroma_selector_cost}")
      if d == 0 and d != self.chroma_depth - 1:
        chroma_selector_cost += 6
      elif d != self.chroma_depth - 1:
        chroma_selector_cost += self.chroma_width # relu
      if self.bias:
        chroma_selector_cost += self.chroma_width
    
    softmax_cost = self.chroma_width * (LOGEXP_COST + DIV_COST + ADD_COST)
    chroma_selector_cost += softmax_cost
    chroma_selector_cost += (upsample_cost * self.chroma_width)
    chroma_selector_cost /= (self.scale_factor*self.scale_factor)

    chroma_interp_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width
      
      chroma_interp_cost += in_c * self.chroma_width * 3 * 3
      if d != self.chroma_depth - 1:
        chroma_interp_cost += self.chroma_width
      if self.bias:
        chroma_interp_cost += self.chroma_width

    chroma_mult_sumR_cost = self.chroma_width # dot product of vector size chroma_width
    chroma_cost = chroma_selector_cost + chroma_interp_cost + chroma_mult_sumR_cost
    print(f"lowres cost {lowres_cost} fullres cost {fullres_cost} multsum r cost {mult_sumR_cost} chroma_cost {chroma_cost}")
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
       
    for l in self.chroma_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.chroma_selector:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

  def forward(self, mosaic):
    # 1/4 resolution features
    mosaic_shape = list(mosaic.shape)

    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(mosaic)

    interps = th.cat((lowres_interp, fullres_interp), 1)
    mul = interps * selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 

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
    rb_min_g = rb - green_rb

    green_shape = [mosaic_shape[0], 4, mosaic_shape[2], mosaic_shape[3]]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,:,:] = mosaic[:,0,:,:]
    green[:,1,:,:] = green_rb[:,0,:,:]
    green[:,2,:,:] = green_rb[:,1,:,:]
    green[:,3,:,:] = mosaic[:,3,:,:]

    chroma_input = th.cat((rb_min_g, green), 1)
    chroma_selector_input = th.cat((rb, green), 1)
    chroma_interps = self.chroma_filter(chroma_input)
    chroma_selector = self.chroma_selector(chroma_input) #self.chroma_selector(chroma_selector_input)

    chroma_weighted = chroma_interps * chroma_selector
    chroma_weighted_reshaped = th.reshape(chroma_weighted, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3]))
    chroma_diff = chroma_weighted_reshaped.sum(2, keepdim=False)

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



"""
Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green.
A third sequence of 3x3 convs on downsampled bayer quad is used to produce mixing weights for
the lowres and fullres interpolations. The first conv is 1x1 to try to learn a change of basis. 

chroma model: 
Weight predictor is on lowres r and b stacked with green predictions, tries to learn a change of basis
in the first conv1x1 layer, then does subsequent series of convs 
Interp is on full res r - g / b - g stacked with the green predictions 
"""
class MultiresQuadRGBV9Model(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=16, bias=False, scale_factor=2):
    super(MultiresQuadRGBV9Model, self).__init__()
    self.width = width
    self.depth = depth
    self.bias = bias
    self.chroma_depth = chroma_depth
    self.chroma_width = chroma_width
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor + 2 # change to scale_factor * 2 

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
    selector_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.depth+1):
      if d == 0:
        in_c = 4
        out_c = 4
        k = 1
      elif d == 1:
        in_c = 4
        out_c = self.width*2
        k = 3
      else:
        in_c = self.width*2
        out_c = self.width*2
        k = 3

      selector_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, k, padding=k//2, bias=self.bias)
      if d != self.depth-1:
        selector_layers[f"relu{d}"] = nn.ReLU()

    selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    selector_layers["softmax"] = nn.Softmax(dim=1)

    self.lowres_filter = nn.Sequential(lowres_layers)
    self.fullres_filter = nn.Sequential(fullres_layers)
    self.selector = nn.Sequential(selector_layers)

    chroma_interp_layers = OrderedDict()
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width

      out_c = self.chroma_width
      chroma_interp_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth - 1:
        chroma_interp_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers = OrderedDict()
    chroma_selector_layers["downsample"] = nn.Conv2d(6, 6, self.downsample_w, stride=scale_factor, padding=1, bias=self.bias)
    for d in range(self.chroma_depth+1):
      if d == 0:
        in_c = 6
        out_c = 6 # try learning change of basis
        k = 1
      elif d == 1:
        in_c = 6
        out_c = self.chroma_width
        k = 3
      else:
        in_c = self.chroma_width
        out_c = self.chroma_width
        k = 3

      chroma_selector_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, k, padding=k//2, bias=self.bias)
      if d != self.depth-1:
        chroma_selector_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    chroma_selector_layers["softmax"] = nn.Softmax(dim=1)

    self.chroma_filter = nn.Sequential(chroma_interp_layers)
    self.chroma_selector = nn.Sequential(chroma_selector_layers)

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
    
    selector_cost = downsample_cost
    for d in range(self.depth+1):
      if d == 0:
        in_c = 4
        out_c = 4
        k = 1
      elif d == 1:
        in_c = 4
        out_c = self.width*2
        k = 3
      else:
        in_c = self.width*2
        out_c = self.width*2
        k = 3

      selector_cost += in_c * out_c * k * k 
      if d != self.depth - 1:
        selector_cost += out_c
      if self.bias:
        selector_cost += out_c

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

    chroma_downsample_cost = 6 * 6 * self.downsample_w * self.downsample_w
    chroma_selector_cost = chroma_downsample_cost
    print(f"chroma selector cost {chroma_selector_cost}")

    for d in range(self.chroma_depth+1):
      if d == 0:
        in_c = 6 # try to learn a change of basis
        out_c = 6
        k = 1
      elif d == 1:
        in_c = 6
        out_c = self.chroma_width
        k = 3
      else:
        in_c = self.chroma_width
        out_c = self.chroma_width
        k = 3

      chroma_selector_cost += in_c * out_c * k * k
      print(f"chroma selector cost {chroma_selector_cost}")
      if d != self.chroma_depth - 1:
        chroma_selector_cost += out_c # relu
    
      if self.bias:
        chroma_selector_cost += out_c
    
    softmax_cost = self.chroma_width * (LOGEXP_COST + DIV_COST + ADD_COST)
    chroma_selector_cost += softmax_cost
    chroma_selector_cost += (upsample_cost * self.chroma_width)
    chroma_selector_cost /= (self.scale_factor*self.scale_factor)

    chroma_interp_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width
      
      chroma_interp_cost += in_c * self.chroma_width * 3 * 3
      if d != self.chroma_depth - 1:
        chroma_interp_cost += self.chroma_width
      if self.bias:
        chroma_interp_cost += self.chroma_width

    chroma_mult_sumR_cost = self.chroma_width * 2
    chroma_cost = chroma_selector_cost + chroma_interp_cost + chroma_mult_sumR_cost
    print(f"lowres cost {lowres_cost} fullres cost {fullres_cost} multsum r cost {mult_sumR_cost} chroma_cost {chroma_cost}")
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
       
    for l in self.chroma_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.chroma_selector:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

  def forward(self, mosaic):
    # 1/4 resolution features
    mosaic_shape = list(mosaic.shape)

    lowres_interp = self.lowres_filter(mosaic)
    fullres_interp = self.fullres_filter(mosaic)
    selectors = self.selector(mosaic)

    interps = th.cat((lowres_interp, fullres_interp), 1)
    mul = interps * selectors

    lowres_mul, fullres_mul = th.split(mul, self.width, dim=1) 
    lmul_reshaped = th.reshape(lowres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3]))  
    fmul_reshaped = th.reshape(fullres_mul, (mosaic_shape[0], 2, self.width//2, mosaic_shape[2], mosaic_shape[3])) 

    weighted_interps = th.cat((lmul_reshaped, fmul_reshaped), 2) 
    green_rb = weighted_interps.sum(2, keepdim=False) 

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
    rb_min_g = rb - green_rb

    green_shape = [mosaic_shape[0], 4, mosaic_shape[2], mosaic_shape[3]]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,:,:] = mosaic[:,0,:,:]
    green[:,1,:,:] = green_rb[:,0,:,:]
    green[:,2,:,:] = green_rb[:,1,:,:]
    green[:,3,:,:] = mosaic[:,3,:,:]

    chroma_input = th.cat((rb_min_g, green), 1)
    chroma_selector_input = th.cat((rb, green), 1)
    chroma_interps = self.chroma_filter(chroma_input)
    chroma_selector = self.chroma_selector(chroma_selector_input)

    chroma_weighted = chroma_interps * chroma_selector
    chroma_weighted_reshaped = th.reshape(chroma_weighted, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3]))
    chroma_diff = chroma_weighted_reshaped.sum(2, keepdim=False)

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

chroma model: 
Weight predictor is on lowres r and b stacked with green predictions, learns a change of basis
in the first conv layer, then does subsequent series of convs 
Interp is on full res r - g / b - g stacked with the green predictions 
"""
class MultiresQuadRGBV10Model(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=12, bias=False, scale_factor=2):
    super(MultiresQuadRGBV10Model, self).__init__()
    self.width = width
    self.depth = depth
    self.chroma_width = chroma_width
    self.chroma_depth = chroma_depth
    self.bias = bias
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor * 2 # change to scale_factor * 2 

    lowres_layers = OrderedDict()
    lowres_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=(self.downsample_w-scale_factor)//2, bias=self.bias)
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

    chroma_interp_layers = OrderedDict()
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width

      out_c = self.chroma_width
      chroma_interp_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth - 1:
        chroma_interp_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers = OrderedDict()
    chroma_selector_layers["downsample"] = nn.Conv2d(6, 6, self.downsample_w, stride=scale_factor, padding=(self.downsample_w-scale_factor)//2, bias=self.bias)
    for d in range(self.chroma_depth+1):
      if d == 0:
        in_c = 6
        out_c = 6 # try learning change of basis
        k = 1
      elif d == 1:
        in_c = 6
        out_c = self.chroma_width
        k = 3
      else:
        in_c = self.chroma_width
        out_c = self.chroma_width
        k = 3

      chroma_selector_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, k, padding=k//2, bias=self.bias)
      if d != self.depth-1:
        chroma_selector_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    chroma_selector_layers["softmax"] = nn.Softmax(dim=1)

    self.chroma_filter = nn.Sequential(chroma_interp_layers)
    self.chroma_selector = nn.Sequential(chroma_selector_layers)

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

    chroma_downsample_cost = 6 * 6 * self.downsample_w * self.downsample_w
    chroma_selector_cost = chroma_downsample_cost
    print(f"chroma selector cost {chroma_selector_cost}")

    for d in range(self.chroma_depth+1):
      if d == 0:
        in_c = 6 # try to learn a change of basis
        out_c = 6
        k = 1
      elif d == 1:
        in_c = 6
        out_c = self.chroma_width
        k = 3
      else:
        in_c = self.chroma_width
        out_c = self.chroma_width
        k = 3

      chroma_selector_cost += in_c * out_c * k * k
      print(f"chroma selector cost {chroma_selector_cost}")
      if d == 0 and d != self.chroma_depth - 1:
        chroma_selector_cost += 6
      elif d != self.chroma_depth - 1:
        chroma_selector_cost += self.chroma_width # relu
      if self.bias:
        chroma_selector_cost += self.chroma_width
    
    softmax_cost = self.chroma_width * (LOGEXP_COST + DIV_COST + ADD_COST)
    chroma_selector_cost += softmax_cost
    chroma_selector_cost += (upsample_cost * self.chroma_width)
    chroma_selector_cost /= (self.scale_factor*self.scale_factor)

    chroma_interp_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width
      
      chroma_interp_cost += in_c * self.chroma_width * 3 * 3
      if d != self.chroma_depth - 1:
        chroma_interp_cost += self.chroma_width
      if self.bias:
        chroma_interp_cost += self.chroma_width

    chroma_mult_sumR_cost = self.chroma_width # dot product of vector size chroma width 

    chroma_cost = chroma_selector_cost + chroma_interp_cost + chroma_mult_sumR_cost
    print(f"lowres cost {lowres_cost} fullres cost {fullres_cost} multsum r cost {mult_sumR_cost} chroma_cost {chroma_cost}")
    cost = (lowres_cost + fullres_cost + selector_cost + mult_sumR_cost + chroma_cost) 
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

    for l in self.chroma_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.chroma_selector:
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
   
    rb_shape = green_rb.shape 
    rb = th.empty(th.Size(rb_shape), device=mosaic.device)
    rb[:,0,:,:] = mosaic[:,1,:,:] 
    rb[:,1,:,:] = mosaic[:,2,:,:] 
    rb_min_g = rb - green_rb

    green_shape = [mosaic_shape[0], 4, mosaic_shape[2], mosaic_shape[3]]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,:,:] = mosaic[:,0,:,:]
    green[:,1,:,:] = green_rb[:,0,:,:]
    green[:,2,:,:] = green_rb[:,1,:,:]
    green[:,3,:,:] = mosaic[:,3,:,:]

    green_add_shape = [mosaic_shape[0], 6, mosaic_shape[2], mosaic_shape[3]]
    green_add = th.empty(th.Size(green_add_shape), device=mosaic.device)
    green_add[:,0,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,1,:,:] = green_rb[:,1,:,:] # B
    green_add[:,2,:,:] = mosaic[:,3,:,:] # GB
    green_add[:,3,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,4,:,:] = green_rb[:,0,:,:] # R
    green_add[:,5,:,:] = mosaic[:,3,:,:] # GB

    chroma_input = th.cat((rb_min_g, green), 1)
    chroma_selector_input = th.cat((rb, green), 1)
    chroma_interps = self.chroma_filter(chroma_input)
    chroma_selector = self.chroma_selector(chroma_selector_input)

    chroma_weighted = chroma_interps * chroma_selector
    chroma_weighted_reshaped = th.reshape(chroma_weighted, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3]))
    chroma_diff = chroma_weighted_reshaped.sum(2, keepdim=False)

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

chroma model: 
Weight predictor is on lowres r and b stacked with green predictions, learns a change of basis
in the first conv layer, then does subsequent series of convs 
Interp is on full res r - g / b - g stacked with the green predictions 
"""
class MultiresQuadRGBV11Model(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=12, bias=False, scale_factor=2):
    super(MultiresQuadRGBV11Model, self).__init__()
    self.width = width
    self.depth = depth
    self.chroma_width = chroma_width
    self.chroma_depth = chroma_depth
    self.bias = bias
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor * 2 # change to scale_factor * 2 

    lowres_layers = OrderedDict()
    lowres_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=(self.downsample_w-scale_factor)//2, bias=self.bias)
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

    chroma_interp_layers = OrderedDict()
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width

      out_c = self.chroma_width
      chroma_interp_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth - 1:
        chroma_interp_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers = OrderedDict()
    chroma_selector_layers["downsample"] = nn.Conv2d(6, 6, self.downsample_w, stride=scale_factor, padding=(self.downsample_w-scale_factor)//2, bias=self.bias)
    for d in range(self.chroma_depth+1):
      if d == 0:
        in_c = 6
        out_c = 6 # try learning change of basis
        k = 1
      elif d == 1:
        in_c = 6
        out_c = self.chroma_width
        k = 3
      else:
        in_c = self.chroma_width
        out_c = self.chroma_width
        k = 3

      chroma_selector_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, k, padding=k//2, bias=self.bias)
      if d != self.depth-1:
        chroma_selector_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    chroma_selector_layers["softmax"] = nn.Softmax(dim=1)

    self.chroma_filter = nn.Sequential(chroma_interp_layers)
    self.chroma_selector = nn.Sequential(chroma_selector_layers)

    self.red_blue_median_filter = nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=self.bias)
    self.green_median_filter = nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=self.bias)

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

    chroma_downsample_cost = 6 * 6 * self.downsample_w * self.downsample_w
    chroma_selector_cost = chroma_downsample_cost
    print(f"chroma selector cost {chroma_selector_cost}")

    for d in range(self.chroma_depth+1):
      if d == 0:
        in_c = 6 # try to learn a change of basis
        out_c = 6
        k = 1
      elif d == 1:
        in_c = 6
        out_c = self.chroma_width
        k = 3
      else:
        in_c = self.chroma_width
        out_c = self.chroma_width
        k = 3

      chroma_selector_cost += in_c * out_c * k * k
      print(f"chroma selector cost {chroma_selector_cost}")
      if d == 0 and d != self.chroma_depth - 1:
        chroma_selector_cost += 6
      elif d != self.chroma_depth - 1:
        chroma_selector_cost += self.chroma_width # relu
      if self.bias:
        chroma_selector_cost += self.chroma_width
    
    softmax_cost = self.chroma_width * (LOGEXP_COST + DIV_COST + ADD_COST)
    chroma_selector_cost += softmax_cost
    chroma_selector_cost += (upsample_cost * self.chroma_width)
    chroma_selector_cost /= (self.scale_factor*self.scale_factor)

    chroma_interp_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width
      
      chroma_interp_cost += in_c * self.chroma_width * 3 * 3
      if d != self.chroma_depth - 1:
        chroma_interp_cost += self.chroma_width
      if self.bias:
        chroma_interp_cost += self.chroma_width

    chroma_mult_sumR_cost = self.chroma_width # dot product of vector size chroma width 

    chroma_cost = chroma_selector_cost + chroma_interp_cost + chroma_mult_sumR_cost
    print(f"lowres cost {lowres_cost} fullres cost {fullres_cost} multsum r cost {mult_sumR_cost} chroma_cost {chroma_cost}")
    cost = (lowres_cost + fullres_cost + selector_cost + mult_sumR_cost + chroma_cost) 
    cost /= 4

    red_blue_median_filter_cost = 3 * 3 * 2 + 2 # two 3x3 convs, subtract and add back green
    green_median_filter_cost = 3 * 3 * 2 + 4 # one 3x3x2 conv, subtract red and blue, add back average between red and blue
    cost += red_blue_median_filter_cost + green_median_filter_cost
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

    for l in self.chroma_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.chroma_selector:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    nn.init.xavier_normal_(self.red_blue_median_filter.weight)
    if self.bias:
      nn.init.zeros_(self.red_blue_median_filter.bias)

    nn.init.xavier_normal_(self.green_median_filter.weight)
    if self.bias:
      nn.init.zeros_(self.green_median_filter.bias)
               

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
   
    rb_shape = green_rb.shape 
    rb = th.empty(th.Size(rb_shape), device=mosaic.device)
    rb[:,0,:,:] = mosaic[:,1,:,:] 
    rb[:,1,:,:] = mosaic[:,2,:,:] 
    rb_min_g = rb - green_rb

    green_shape = [mosaic_shape[0], 4, mosaic_shape[2], mosaic_shape[3]]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,:,:] = mosaic[:,0,:,:]
    green[:,1,:,:] = green_rb[:,0,:,:]
    green[:,2,:,:] = green_rb[:,1,:,:]
    green[:,3,:,:] = mosaic[:,3,:,:]

    green_add_shape = [mosaic_shape[0], 6, mosaic_shape[2], mosaic_shape[3]]
    green_add = th.empty(th.Size(green_add_shape), device=mosaic.device)
    green_add[:,0,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,1,:,:] = green_rb[:,1,:,:] # B
    green_add[:,2,:,:] = mosaic[:,3,:,:] # GB
    green_add[:,3,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,4,:,:] = green_rb[:,0,:,:] # R
    green_add[:,5,:,:] = mosaic[:,3,:,:] # GB

    chroma_input = th.cat((rb_min_g, green), 1)
    chroma_selector_input = th.cat((rb, green), 1)
    chroma_interps = self.chroma_filter(chroma_input)
    chroma_selector = self.chroma_selector(chroma_selector_input)

    chroma_weighted = chroma_interps * chroma_selector
    chroma_weighted_reshaped = th.reshape(chroma_weighted, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3]))
    chroma_diff = chroma_weighted_reshaped.sum(2, keepdim=False)

    chroma_pred = chroma_diff + green_add

    # median filtering 
    # R - G, B - G, G - B, G - R
    one_chan_shape = [mosaic_shape[0], 1, mosaic_shape[2]*2, mosaic_shape[3]*2]
    red = th.empty(th.Size(one_chan_shape), device=mosaic.device)
    blue = th.empty(th.Size(one_chan_shape), device=mosaic.device)
    green = th.empty(th.Size(one_chan_shape), device=mosaic.device)

    red[:,0,0::2,0::2] = chroma_pred[:,0,:,:]
    red[:,0,0::2,1::2] = mosaic[:,1,:,:]
    red[:,0,1::2,0::2] = chroma_pred[:,1,:,:]
    red[:,0,1::2,1::2] = chroma_pred[:,2,:,:]

    green[:,0,0::2,0::2] = mosaic[:,0,:,:]
    green[:,0,0::2,1::2] = green_rb[:,0,:,:]
    green[:,0,1::2,0::2] = green_rb[:,1,:,:]
    green[:,0,1::2,1::2] = mosaic[:,3,:,:]

    blue[:,0,0::2,0::2] = chroma_pred[:,3,:,:]
    blue[:,0,0::2,1::2] = chroma_pred[:,4,:,:]
    blue[:,0,1::2,0::2] = mosaic[:,2,:,:]
    blue[:,0,1::2,1::2] = chroma_pred[:,5,:,:]

    red_minus_green = red - green
    blue_minus_green = blue - green
    green_minus_red = green - red
    green_minus_blue = green - blue

    red_blue_stack = th.cat((red_minus_green, blue_minus_green), 1)
    green_stack = th.cat((green_minus_red, green_minus_blue), 1)

    red_blue_median = self.red_blue_median_filter(red_blue_stack) + green
    green_median = self.green_median_filter(green_stack) + th.cat((red, blue), 1)
    final_green = green_median.sum(1, keepdim=True) * 0.5
    final_red, final_blue = th.split(red_blue_median, 1, dim=1)
    img = th.cat((final_red, final_green, final_blue), 1)

    return img


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

chroma model: 
Weight predictor is on lowres r and b stacked with green predictions, learns a change of basis
in the first conv layer, then does subsequent series of convs 
Interp is on full res r - g / b - g stacked with the green predictions 
"""
class MultiresQuadRGBV12Model(nn.Module):
  def __init__(self, width=12, depth=2, chroma_depth=1, chroma_width=12, bias=False, scale_factor=2):
    super(MultiresQuadRGBV12Model, self).__init__()
    self.width = width
    self.depth = depth
    self.chroma_width = chroma_width
    self.chroma_depth = chroma_depth
    self.bias = bias
    self.scale_factor = scale_factor
    # low res 
    self.downsample_w = scale_factor * 2 # change to scale_factor * 2 

    lowres_layers = OrderedDict()
    lowres_layers["downsample"] = nn.Conv2d(4, 4, self.downsample_w, stride=scale_factor, padding=(self.downsample_w-scale_factor)//2, bias=self.bias)
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

    chroma_interp_layers = OrderedDict()
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width

      out_c = self.chroma_width
      chroma_interp_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, 3, padding=1, bias=self.bias)
      if d != self.chroma_depth - 1:
        chroma_interp_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers = OrderedDict()
    chroma_selector_layers["downsample"] = nn.Conv2d(6, 6, self.downsample_w, stride=scale_factor, padding=(self.downsample_w-scale_factor)//2, bias=self.bias)
    for d in range(self.chroma_depth+1):
      if d == 0:
        in_c = 6
        out_c = 6 # try learning change of basis
        k = 1
      elif d == 1:
        in_c = 6
        out_c = self.chroma_width
        k = 3
      else:
        in_c = self.chroma_width
        out_c = self.chroma_width
        k = 3

      chroma_selector_layers[f"conv{d}"] = nn.Conv2d(in_c, out_c, k, padding=k//2, bias=self.bias)
      if d != self.depth-1:
        chroma_selector_layers[f"relu{d}"] = nn.ReLU()

    chroma_selector_layers["upsample"] = nn.Upsample(scale_factor=scale_factor, mode='bilinear') #nn.ConvTranspose2d(width*2, width*2, 2, stride=2, groups=width*2, bias=False)
    chroma_selector_layers["softmax"] = nn.Softmax(dim=1)

    self.chroma_filter = nn.Sequential(chroma_interp_layers)
    self.chroma_selector = nn.Sequential(chroma_selector_layers)

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

    chroma_downsample_cost = 6 * 6 * self.downsample_w * self.downsample_w
    chroma_selector_cost = chroma_downsample_cost
    print(f"chroma selector cost {chroma_selector_cost}")

    for d in range(self.chroma_depth+1):
      if d == 0:
        in_c = 6 # try to learn a change of basis
        out_c = 6
        k = 1
      elif d == 1:
        in_c = 6
        out_c = self.chroma_width
        k = 3
      else:
        in_c = self.chroma_width
        out_c = self.chroma_width
        k = 3

      chroma_selector_cost += in_c * out_c * k * k
      print(f"chroma selector cost {chroma_selector_cost}")
      if d == 0 and d != self.chroma_depth - 1:
        chroma_selector_cost += 6
      elif d != self.chroma_depth - 1:
        chroma_selector_cost += self.chroma_width # relu
      if self.bias:
        chroma_selector_cost += self.chroma_width
    
    softmax_cost = self.chroma_width * (LOGEXP_COST + DIV_COST + ADD_COST)
    chroma_selector_cost += softmax_cost
    chroma_selector_cost += (upsample_cost * self.chroma_width)
    chroma_selector_cost /= (self.scale_factor*self.scale_factor)

    chroma_interp_cost = 0
    for d in range(self.chroma_depth):
      if d == 0:
        in_c = 6
      else:
        in_c = self.chroma_width
      
      chroma_interp_cost += in_c * self.chroma_width * 3 * 3
      if d != self.chroma_depth - 1:
        chroma_interp_cost += self.chroma_width
      if self.bias:
        chroma_interp_cost += self.chroma_width

    chroma_mult_sumR_cost = self.chroma_width # dot product of vector size chroma width 

    chroma_cost = chroma_selector_cost + chroma_interp_cost + chroma_mult_sumR_cost
    print(f"lowres cost {lowres_cost} fullres cost {fullres_cost} multsum r cost {mult_sumR_cost} chroma_cost {chroma_cost}")
    cost = (lowres_cost + fullres_cost + selector_cost + mult_sumR_cost + chroma_cost) 
    cost /= 4

    MEDIAN_COST = 30 
    red_blue_median_filter_cost = (MEDIAN_COST + 2) * 2
    green_median_filter_cost = MEDIAN_COST + 6
    cost += red_blue_median_filter_cost + green_median_filter_cost
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

    for l in self.chroma_filter:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

    for l in self.chroma_selector:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)


  def forward(self, mosaic):
    # 1/4 resolution features
    mosaic_shape = list(mosaic.shape)
    n = mosaic_shape[0]

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
   
    rb_shape = green_rb.shape 
    rb = th.empty(th.Size(rb_shape), device=mosaic.device)
    rb[:,0,:,:] = mosaic[:,1,:,:] 
    rb[:,1,:,:] = mosaic[:,2,:,:] 
    rb_min_g = rb - green_rb

    green_shape = [mosaic_shape[0], 4, mosaic_shape[2], mosaic_shape[3]]
    green = th.empty(th.Size(green_shape), device=mosaic.device)
    green[:,0,:,:] = mosaic[:,0,:,:]
    green[:,1,:,:] = green_rb[:,0,:,:]
    green[:,2,:,:] = green_rb[:,1,:,:]
    green[:,3,:,:] = mosaic[:,3,:,:]

    green_add_shape = [mosaic_shape[0], 6, mosaic_shape[2], mosaic_shape[3]]
    green_add = th.empty(th.Size(green_add_shape), device=mosaic.device)
    green_add[:,0,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,1,:,:] = green_rb[:,1,:,:] # B
    green_add[:,2,:,:] = mosaic[:,3,:,:] # GB
    green_add[:,3,:,:] = mosaic[:,0,:,:]  # GR
    green_add[:,4,:,:] = green_rb[:,0,:,:] # R
    green_add[:,5,:,:] = mosaic[:,3,:,:] # GB

    chroma_input = th.cat((rb_min_g, green), 1)
    chroma_selector_input = th.cat((rb, green), 1)
    chroma_interps = self.chroma_filter(chroma_input)
    chroma_selector = self.chroma_selector(chroma_selector_input)

    chroma_weighted = chroma_interps * chroma_selector
    chroma_weighted_reshaped = th.reshape(chroma_weighted, (mosaic_shape[0], 6, self.chroma_width//6, mosaic_shape[2], mosaic_shape[3]))
    chroma_diff = chroma_weighted_reshaped.sum(2, keepdim=False)

    chroma_pred = chroma_diff + green_add

    # median filtering 
    # R - G, B - G, G - B, G - R
    h = mosaic_shape[2]*2
    w = mosaic_shape[3]*2
    one_chan_shape = [mosaic_shape[0], 1, h, w]
    red = th.empty(th.Size(one_chan_shape), device=mosaic.device)
    blue = th.empty(th.Size(one_chan_shape), device=mosaic.device)
    green = th.empty(th.Size(one_chan_shape), device=mosaic.device)

    red[:,0,0::2,0::2] = chroma_pred[:,0,:,:]
    red[:,0,0::2,1::2] = mosaic[:,1,:,:]
    red[:,0,1::2,0::2] = chroma_pred[:,1,:,:]
    red[:,0,1::2,1::2] = chroma_pred[:,2,:,:]

    green[:,0,0::2,0::2] = mosaic[:,0,:,:]
    green[:,0,0::2,1::2] = green_rb[:,0,:,:]
    green[:,0,1::2,0::2] = green_rb[:,1,:,:]
    green[:,0,1::2,1::2] = mosaic[:,3,:,:]

    blue[:,0,0::2,0::2] = chroma_pred[:,3,:,:]
    blue[:,0,0::2,1::2] = chroma_pred[:,4,:,:]
    blue[:,0,1::2,0::2] = mosaic[:,2,:,:]
    blue[:,0,1::2,1::2] = chroma_pred[:,5,:,:]

    red_minus_green = red - green
    blue_minus_green = blue - green
    green_minus_red = green - red
    green_minus_blue = green - blue

    red_min_g_unfold = th.nn.functional.unfold(red_minus_green, 3, dilation=1, padding=1, stride=1).reshape(n,1,9,-1)
    blue_min_g_unfold = th.nn.functional.unfold(blue_minus_green, 3, dilation=1, padding=1, stride=1).reshape(n,1,9,-1)
    green_min_r_unfold = th.nn.functional.unfold(green_minus_red, 3, dilation=1, padding=1, stride=1).reshape(n,1,9,-1)
    green_min_b_unfold = th.nn.functional.unfold(green_minus_blue, 3, dilation=1, padding=1, stride=1).reshape(n,1,9,-1)
    red_min_g_med, _ = th.median(red_min_g_unfold, dim=2, keepdim=True)
    blue_min_g_med, _ = th.median(blue_min_g_unfold, dim=2, keepdim=True)
    green_min_r_med, _ = th.median(green_min_r_unfold, dim=2, keepdim=True)
    green_min_b_med, _ = th.median(green_min_b_unfold, dim=2, keepdim=True)

    red_median = red_min_g_med.reshape(n,1,h,w) + green
    blue_median = blue_min_g_med.reshape(n,1,h,w) + green
    green_r_median = green_min_r_med.reshape(n,1,h,w) + red 
    green_b_median = green_min_b_med.reshape(n,1,h,w) + blue
    green_median = (green_r_median + green_b_median) * 0.5

    img = th.cat((red_median, green_median, blue_median), 1)

    return img


"""
Takes green prediction as input and predicts Chroma
"""
class MultiresQuadRBModel(nn.Module):
  def __init__(self, width=12, chroma_depth=1, bias=False):
    super(MultiresQuadRBModel, self).__init__()
    self.width = width
    self.chroma_depth = chroma_depth
    self.bias = bias
    
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

    cost = chroma_cost
    cost /= 4

    return cost

  def _initialize_parameters(self):      
    for l in self.chroma_processor:
      print(l)
      if hasattr(l, "weight"):
        print(l.__class__)
        nn.init.xavier_normal_(l.weight)
        if self.bias:
          nn.init.zeros_(l.bias)

  def forward(self, mosaic, green):
    mosaic_shape = list(mosaic.shape)

    green_rb_shape = [mosaic_shape[0], 2, mosaic_shape[2], mosaic_shape[3]]
    green_rb = th.empty(th.Size(green_rb_shape), device=mosaic.device)
    green_rb[:,0,:,:] = green[:,1,:,:]
    green_rb[:,1,:,:] = green[:,2,:,:]

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

