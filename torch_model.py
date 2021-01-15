"""
      ieturn out
lowered representation of AST to pytorch model
"""

import torch 
import torch.nn as nn
from demosaic_ast import *
from config import IMG_H, IMG_W

# from a github gist by victorlei
def extclass(cls):
  return lambda f: (setattr(cls,f.__name__,f) or f)



class InputOp(nn.Module):
  def __init__(self, input_name, model=None, model_name=None, no_grad=False, weights=None):
    super(InputOp, self).__init__()
    self.name = input_name
    if model:
      self.model = model
      self.model_name = f"input_model_{model_name}"
      setattr(self, self.model_name, model)
      self.no_grad = no_grad 
      self.weights = weights
      self.initialized = False

    self.output = None

  def _initialize_parameters(self):
    if hasattr(self, "model") and not self.initialized:
      if self.weights:
        print(f"loading pretrained weights for input model {self.name} no grad {self.no_grad}")
        state_dict = torch.load(self.weights)
        self.model.load_state_dict(state_dict)
      else:
        print(f"initializing weight for input model {self.name} no grad {self.no_grad}")
        self.model._initialize_parameters()

      self.initialized = True

  def reset(self):
    self.output = None
    if hasattr(self, "model"):
      self.model.reset()

  # unnecessary function, never called but needed by nn.Module
  def foward(self, model_inputs):
    return model_inputs[self.name]

  def run(self, model_inputs):
    if self.output is None:
      if hasattr(self, "model"):
        if not self.model.output is None:
          self.output = self.model.output
          return self.output
        else:
          if self.no_grad:
            with torch.no_grad():
              self.output = self.model.run(model_inputs)
          else:
            self.output = self.model.run(model_inputs)
          return self.output
      else:
        return model_inputs[self.name]
    else:
      if hasattr(self, "model"):
        return self.output 
      else:
        return model_inputs[self.name]

  def to_gpu(self, gpu_id):
    if hasattr(self, "model"):
      self.model.to_gpu(gpu_id)


class AddOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(AddOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      loperand = self._operands[0].run(model_inputs)
      roperand = self._operands[1].run(model_inputs)
      self.output = self.forward(loperand, roperand)
    return self.output 

  def forward(self, x, y):
    # for handling quad layouts that can't use default pytorch broadcasting:
    # if channel counts not the same and smaller count != 1 --> tile tensor
    # the larger channel count should be a multiple of the smaller
    xc = x.shape[1]
    yc = y.shape[1]
    if xc != yc and min(xc, yc) != 1:
      if xc < yc:
        factor = yc // xc
        x = x.repeat(1, factor, 1, 1)
      else:
        factor = xc // yc 
        y = y.repeat(1, factor, 1, 1)
    return x + y

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)


class SubOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(SubOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      loperand = self._operands[0].run(model_inputs)
      roperand = self._operands[1].run(model_inputs)
      self.output = self.forward(loperand, roperand)
    return self.output 

  def forward(self, x, y): 
    # for handling quad layouts that can't use default pytorch broadcasting:
    # if channel counts not the same and smaller count != 1 --> tile tensor
    # the larger channel count should be a multiple of the smaller
    xc = x.shape[1]
    yc = y.shape[1]
    if xc != yc and min(xc, yc) != 1:
      if xc < yc:
        factor = yc // xc
        x = x.repeat(1, factor, 1, 1)
      else:
        factor = xc // yc 
        y = y.repeat(1, factor, 1, 1)
    return x - y

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)
    

class MulOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(MulOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      loperand = self._operands[0].run(model_inputs)
      roperand = self._operands[1].run(model_inputs)
      self.output = self.forward(loperand, roperand)
    return self.output 

  def forward(self, x, y):
    # for handling quad layouts that can't use default pytorch broadcasting:
    # if channel counts not the same and smaller count != 1 --> tile tensor
    # the larger channel count should be a multiple of the smaller
    xc = x.shape[1]
    yc = y.shape[1]
    if xc != yc and min(xc, yc) != 1:
      if xc < yc:
        factor = yc // xc
        x = x.repeat(1, factor, 1, 1)
      else:
        factor = xc // yc 
        y = y.repeat(1, factor, 1, 1)
    return x * y

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)


class LogSubOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(LogSubOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      loperand = self._operands[0].run(model_inputs)
      roperand = self._operands[1].run(model_inputs)
      self.output = self.forward(loperand, roperand)
    return self.output 

  def forward(self, x, y):
    # for handling quad layouts that can't use default pytorch broadcasting:
    # if channel counts not the same and smaller count != 1 --> tile tensor
    # the larger channel count should be a multiple of the smaller
    xc = x.shape[1]
    yc = y.shape[1]
    if xc != yc and min(xc, yc) != 1:
      if xc < yc:
        factor = yc // xc
        x = x.repeat(1, factor, 1, 1)
      else:
        factor = xc // yc 
        y = y.repeat(1, factor, 1, 1)
    return torch.log(x) - torch.log(y)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)


class AddExpOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(AddExpOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      loperand = self._operands[0].run(model_inputs)
      roperand = self._operands[1].run(model_inputs)
      self.output = self.forward(loperand, roperand)
    return self.output

  def forward(self, x, y):
    # for handling quad layouts that can't use default pytorch broadcasting:
    # if channel counts not the same and smaller count != 1 --> tile tensor
    # the larger channel count should be a multiple of the smaller
    xc = x.shape[1]
    yc = y.shape[1]
    if xc != yc and min(xc, yc) != 1:
      if xc < yc:
        factor = yc // xc
        x = x.repeat(1, factor, 1, 1)
      else:
        factor = xc // yc 
        y = y.repeat(1, factor, 1, 1)
    return torch.exp(x) + torch.exp(y)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)


class StackOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(StackOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      loperand = self._operands[0].run(model_inputs)
      roperand = self._operands[1].run(model_inputs)
      self.output = self.forward(loperand, roperand)
    return self.output 

  def forward(self, x, y):
    return torch.cat((x, y), 1)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)


# Takes FullGreenQuad predction, RedBlueQuad from Bayer, and 
# 6 channel chroma quad prection: R@Gr, B@R, R@B, R@Gb, B@Gr, B@Gb
# spits out full RGB Image at full resolution
class RGBExtractorOp(nn.Module):
  def __init__(self, fullgreen, redbluebayer, chromapred):
    super(RGBExtractorOp, self).__init__()
    self._operands = nn.ModuleList([fullgreen, redbluebayer, chromapred])
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()
    self._operands[2]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()
    self._operands[2].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand1 = self._operands[0].run(model_inputs)
      operand2 = self._operands[1].run(model_inputs)
      operand3 = self._operands[2].run(model_inputs)
      self.output = self.forward(operand1, operand2, operand3)
    return self.output 

  def forward(self, fullgreen, redbluebayer, chromapred):
    # fullgreen : 4 channels
    # redbluebayer : 2 channels  
    # chromapred: 6 channels R@Gr, B@R, R@B, R@Gb, B@Gr, B@Gb
    fullgreen_shape = list(fullgreen.shape)
    out_shape = [fullgreen_shape[0], 3, fullgreen_shape[2]*2, fullgreen_shape[3]*2]
    img = torch.empty(torch.Size(out_shape), device=fullgreen.device)

    # fill in reds
    img[:,0,0::2,0::2] = chromapred[:,0,:,:]
    img[:,0,0::2,1::2] = redbluebayer[:,0,:,:]
    img[:,0,1::2,0::2] = chromapred[:,2,:,:]
    img[:,0,1::2,1::2] = chromapred[:,3,:,:]

    # fill in greens
    img[:,1,:,:] = self.pixel_shuffle(fullgreen)[:,0,:,:]
   
    # fill in blues
    img[:,2,0::2,0::2] = chromapred[:,4,:,:]
    img[:,2,0::2,1::2] = chromapred[:,1,:,:]
    img[:,2,1::2,0::2] = redbluebayer[:,1,:,:]
    img[:,2,1::2,1::2] = chromapred[:,5,:,:]
    
    return img 

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)
    self._operands[2].to_gpu(gpu_id)


class RGB8ChanExtractorOp(nn.Module):
  def __init__(self, operand1, operand2):
    super(RGB8ChanExtractorOp, self).__init__()
    self._operands = nn.ModuleList([operand1, operand2])
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand1 = self._operands[0].run(model_inputs)
      operand2 = self._operands[1].run(model_inputs)
      self.output = self.forward(operand1, operand2)
    return self.output

  """
  Takes bayer quad and 2 channel green prediction
  Spits out full green channel 
  """
  def forward(self, bayer_quad, rgb_pred):
    # input: bayer an rgb missing values
    # output: full image
    bayer_quad_shape = list(bayer_quad.shape)
    N = bayer_quad_shape[0]
    img_h = bayer_quad_shape[2] * 2
    img_w = img_h
    out_shape = [N, 3, img_h, img_w]

    flat_bayer = self.pixel_shuffle(bayer_quad)
    flat_pred = self.pixel_shuffle(rgb_pred)

    output = torch.empty(torch.Size(out_shape), device=bayer_quad.device)
    
    output[:,0,0::2,0::2] =  flat_pred[:,0,0::2,0::2] # R at Gr
    output[:,1,0::2,0::2] = flat_bayer[:,0,0::2,0::2]
    output[:,2,0::2,0::2] =  flat_pred[:,1,0::2,0::2] # B at Gr

    output[:,0,0::2,1::2] = flat_bayer[:,0,0::2,1::2] 
    output[:,1,0::2,1::2] =  flat_pred[:,0,0::2,1::2] # G at R
    output[:,2,0::2,1::2] =  flat_pred[:,1,0::2,1::2] # B at R

    output[:,0,1::2,0::2] =  flat_pred[:,0,1::2,0::2] # R at B
    output[:,1,1::2,0::2] =  flat_pred[:,1,1::2,0::2] # G at B
    output[:,2,1::2,0::2] = flat_bayer[:,0,1::2,0::2] 

    output[:,0,1::2,1::2] =  flat_pred[:,0,1::2,1::2] # R at Gb
    output[:,1,1::2,1::2] = flat_bayer[:,0,1::2,1::2] 
    output[:,2,1::2,1::2] =  flat_pred[:,1,1::2,1::2] # B at Gb

    return output

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)


class GreenExtractorOp(nn.Module):
  def __init__(self, operand1, operand2):
    super(GreenExtractorOp, self).__init__()
    self._operands = nn.ModuleList([operand1, operand2])
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand1 = self._operands[0].run(model_inputs)
      operand2 = self._operands[1].run(model_inputs)
      self.output = self.forward(operand1, operand2)
    return self.output

  """
  Takes bayer quad and 2 channel green prediction
  Spits out full green channel 
  """
  def forward(self, bayer_quad, green_pred):
    # input: quad predictions and bayer 
    # output: green channel
    out = self.pixel_shuffle(bayer_quad)
    out[:,0,0::2,1::2] = green_pred[:,0,:,:]
    out[:,0,1::2,0::2] = green_pred[:,1,:,:]
    return out

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)


"""
takes flat green prediction and extracts out 2 channel
green predicted values at Red and Blue bayer quad locations
"""
class GreenRBExtractorOp(nn.Module):
  def __init__(self, operand):
    super(GreenRBExtractorOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, flat_green):
    # input: flat green channel 
    # output: green at Red and Blue
    flat_green_shape = list(flat_green.shape)
    N = flat_green_shape[0]
    quad_h = flat_green_shape[2] // 2
    quad_w = quad_h
    out_shape = [N, 2, quad_h, quad_w]

    green_quad = torch.empty(torch.Size(out_shape), device=flat_green.device)
    green_quad[:,0,:,:] = flat_green[:,0,0::2,1::2]
    green_quad[:,1,:,:] = flat_green[:,0,1::2,0::2]
   
    return green_quad

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


"""
takes flat channel prediction and returns full 
channel in bayer quad format
"""
class Flat2QuadOp(nn.Module):
  def __init__(self, operand):
    super(Flat2QuadOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, flat):
    # input: flat green channel 
    # output: green at Red and Blue
    flat_shape = list(flat.shape)
    N = flat_shape[0]
    quad_h = flat_shape[2] // 2
    quad_w = quad_h
    out_shape = [N, 4, quad_h, quad_w]

    quad = torch.empty(torch.Size(out_shape), device=flat.device)
    quad[:,0,:,:] = flat[:,0,0::2,0::2]
    quad[:,1,:,:] = flat[:,0,0::2,1::2]
    quad[:,2,:,:] = flat[:,0,1::2,0::2]
    quad[:,3,:,:] = flat[:,0,1::2,1::2]

    return quad

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class SoftmaxOp(nn.Module):
  def __init__(self, operand):
    super(SoftmaxOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.f = nn.Softmax(dim=1)
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, x):
    return self.f(x)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class ReluOp(nn.Module):
  def __init__(self, operand):
    super(ReluOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.f = nn.ReLU()
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, x):
    return self.f(x)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class LogOp(nn.Module):
  def __init__(self, operand):
    super(LogOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, x):
    return torch.log(x)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class ExpOp(nn.Module):
  def __init__(self, operand):
    super(ExpOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, x):
    return torch.exp(x)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class DownsampleOp(nn.Module):
  def __init__(self, operand, C_in, scale_factor, param_name):
    super(DownsampleOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.scale_factor = scale_factor
    self.downsample_w = scale_factor * 2

    downsampler = nn.Conv2d(C_in, C_in, self.downsample_w, stride=scale_factor, padding=(self.downsample_w-self.scale_factor)//2, bias=False)
    self.param_name = param_name
    setattr(self, param_name, downsampler)
    self.output = None

  def _initialize_parameters(self):
    downsampler = getattr(self, self.param_name)
    nn.init.xavier_normal_(downsampler.weight)
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, x):
    downsampler = getattr(self, self.param_name)
    return downsampler(x)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class UpsampleOp(nn.Module):
  def __init__(self, operand, C_in, scale_factor, param_name):
    super(UpsampleOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.in_c = C_in
    self.scale_factor = scale_factor
    self.param_name = param_name
    bilinear = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
    setattr(self, self.param_name, bilinear)
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, x):
    upsampler = getattr(self, self.param_name)
    return upsampler(x)
    
  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class Conv1x1Op(nn.Module):
  def __init__(self, operand, C_in, C_out, groups, param_name):
    super(Conv1x1Op, self).__init__()
    self._operands = nn.ModuleList([operand])
    f = nn.Conv2d(C_in, C_out, (1, 1), groups=groups, bias=False, padding=0)
    self.param_name = param_name
    setattr(self, param_name, f)
    self.output = None

  def _initialize_parameters(self):
    f = getattr(self, self.param_name)
    nn.init.xavier_normal_(f.weight)
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, x): 
    f = getattr(self, self.param_name)
    out = f(x)
    return out

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


# 1D diagonal convolution from top left corner to bottom right corner
class DiagLRConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, padding, param_name):
    super(DiagLRConv, self).__init__()
    self.padding = padding
    filter_w = nn.Parameter(torch.zeros(C_out, C_in, kernel_size))
    self.param_name = param_name
    setattr(self, param_name, filter_w)

  def _initialize_parameters(self):
    filter_w = getattr(self, self.param_name)
    nn.init.xavier_normal_(filter_w)

  def forward(self, x):
    padding = self.padding
    filter_w = getattr(self, self.param_name)
    return nn.functional.conv2d(x, torch.diag_embed(filter_w), padding=padding)

  def to_gpu(self, gpu_id):
    pass

# 1D diagonal convolution from top right corner to bottom left corner
class DiagRLConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, padding, param_name):
    super(DiagRLConv, self).__init__()
    self.padding = padding
    # self.mask = torch.zeros(C_out, C_in, kernel_size, kernel_size).cuda()
    filter_w = nn.Parameter(torch.zeros(C_out, C_in, kernel_size, kernel_size))
    self.param_name = param_name
    setattr(self, param_name, filter_w)
    self.mask = torch.zeros(C_out, C_in, kernel_size, kernel_size)
   
    for i in range(kernel_size):
      self.mask[..., i, kernel_size-i-1] = 1.0

  def _initialize_parameters(self):
    filter_w = getattr(self, self.param_name)
    nn.init.xavier_normal_(filter_w)

  def forward(self, x):
    filter_w = getattr(self, self.param_name)
    # if filter_w.is_cuda and not self.mask.is_cuda:
    #   self.mask = self.mask.to(device=f"cuda:{self.gpu_id}")
    return nn.functional.conv2d(x, (filter_w * self.mask), padding=self.padding)

  def to_gpu(self, gpu_id):
    self.mask = self.mask.to(device=f"cuda:{gpu_id}")


class Conv1DOp(nn.Module):
  def __init__(self, operand, C_in, C_out, groups, param_name, kwidth):
    super(Conv1DOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.param_name_v = f"{param_name}_v"
    self.param_name_h = f"{param_name}_h"

    num_vfilters = C_out // 2
    num_hfilters = C_out - num_vfilters
    v = nn.Conv2d(C_in, num_vfilters, (kwidth, 1), groups=groups, bias=False, padding=(kwidth//2, 0))
    h = nn.Conv2d(C_in, num_hfilters, (1, kwidth), groups=groups, bias=False, padding=(0, kwidth//2))
  
    setattr(self, self.param_name_v, v)
    setattr(self, self.param_name_h, h)
    self.output = None

  def _initialize_parameters(self):
    v = getattr(self, self.param_name_v)
    h = getattr(self, self.param_name_h)

    nn.init.xavier_normal_(v.weight)
    nn.init.xavier_normal_(h.weight)   
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, x):
    v = getattr(self, self.param_name_v)
    h = getattr(self, self.param_name_h)
    return torch.cat((v(x), h(x)), 1)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class Conv2DOp(nn.Module):
  def __init__(self, operand, C_in, C_out, groups, param_name, kwidth):
    super(Conv2DOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.param_name = param_name
    f = nn.Conv2d(C_in, C_out, (kwidth,kwidth), groups=groups, bias=False, padding=kwidth//2)
    setattr(self, self.param_name, f)
    self.output = None

  def _initialize_parameters(self):
    f = getattr(self, self.param_name)
    nn.init.xavier_normal_(f.weight)
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 
  def forward(self, x):
    f = getattr(self, self.param_name)
    return f(x)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class GroupedSumOp(nn.Module):
  def __init__(self, operand, C_out):
    super(GroupedSumOp, self).__init__()
    self.C_out = C_out
    self._operands = nn.ModuleList([operand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, x):
    x_shape = list(x.shape)
    x_reshaped = torch.reshape(x, (x_shape[0], self.C_out, x_shape[1]//self.C_out, x_shape[2], x_shape[3]))
    out = x_reshaped.sum(2, keepdim=False) 
    return out

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class InterleavedSumOp(nn.Module):
  def __init__(self, operand, C_out):
    super(InterleavedSumOp, self).__init__()
    self.C_out = C_out
    self._operands = nn.ModuleList([operand])
    self.output = None

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self.output = None

  def run(self, model_inputs):
    if self.output is None:
      operand = self._operands[0].run(model_inputs)
      self.output = self.forward(operand)
    return self.output 

  def forward(self, x):
    x_shape = list(x.shape)
    x_reshaped = torch.reshape(x, (x_shape[0], x_shape[1]//self.C_out, self.C_out, x_shape[2], x_shape[3]))
    out = x_reshaped.sum(1, keepdim=False) 
    return out
    
  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


"""
Collects all the operators in a DAG
"""
def collect_operators(model, operators=None, seen=None):
  if operators is None:
    operators = []
  if seen is None:
    seen = set()

  if not(id(model)) in seen:
    seen.add(id(model))
    operators += [model]

  if hasattr(model, "_operands"):
    if len(model._operands) == 3:
      collect_operators(model._operands[0], operators, seen)
      collect_operators(model._operands[1], operators, seen)
      collect_operators(model._operands[2], operators, seen)
    elif len(model._operands) == 2:
      collect_operators(model._operands[0], operators, seen)
      collect_operators(model._operands[1], operators, seen)
    elif len(model._operands) == 1:
      collect_operators(model._operands[0], operators, seen)
    return operators


@extclass(Input)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  if hasattr(self, "node"):
    if id(self.node) in shared_children:
      node_model = shared_children[id(self.node)]
    else:
      node_model = self.node.ast_to_model(shared_children)
      shared_children[id(self.node)] = node_model

    if hasattr(self, "weight_file"):
      input_op = InputOp(self.name, model=node_model, model_name=self.name, no_grad=self.no_grad, weights=self.weight_file)
    else:
      input_op = InputOp(self.name, model=node_model, model_name=self.name, no_grad=self.no_grad)      
  else:
    input_op = InputOp(self.name)

  shared_children[id(self)] = input_op 
  return input_op

@extclass(Add)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  model = AddOp(lmodel, rmodel)

  shared_children[id(self)] = model 
  return model

@extclass(Sub)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  model = SubOp(lmodel, rmodel)

  shared_children[id(self)] = model 
  return model

@extclass(Mul)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  model = MulOp(lmodel, rmodel)

  shared_children[id(self)] = model 
  return model

@extclass(LogSub)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  lmodel = self.lchild.ast_to_model(shared_children)
  assert (type(self.rchild.parent) is tuple), "Right child of LogSub should have two parents"
  if id(self.rchild) in shared_children:
    rmodel = shared_children[id(self.rchild)]
  else:
    rmodel = self.rchild.ast_to_model(shared_children)
    shared_children[id(self.rchild)] = rmodel
  return LogSubOp(lmodel, rmodel)

@extclass(AddExp)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  lmodel = self.lchild.ast_to_model(shared_children)
  assert (type(self.rchild.parent) is tuple), "Right child of AddExp should have two parents"
  if id(self.rchild) in shared_children:
    rmodel = shared_children[id(self.rchild)]
  else:
    rmodel = self.rchild.ast_to_model(shared_children)
    shared_children[id(self.rchild)] = rmodel
  return AddExpOp(lmodel, rmodel)

@extclass(Stack)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  model = StackOp(lmodel, rmodel)

  shared_children[id(self)] = model 
  return model

@extclass(RGBExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child1_model = self.child1.ast_to_model(shared_children)
  child2_model = self.child2.ast_to_model(shared_children)
  child3_model = self.child3.ast_to_model(shared_children)
  model = RGBExtractorOp(child1_model, child2_model, child3_model)

  shared_children[id(self)] = model 
  return model

@extclass(RGB8ChanExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  model = RGB8ChanExtractorOp(lmodel, rmodel)
  
  shared_children[id(self)] = model 
  return model

@extclass(GreenExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  model = GreenExtractorOp(lmodel, rmodel)
  
  shared_children[id(self)] = model 
  return model

@extclass(GreenRBExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = GreenRBExtractorOp(child_model)

  shared_children[id(self)] = model 
  return model

@extclass(Flat2Quad)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = Flat2QuadOp(child_model)

  shared_children[id(self)] = model 
  return model

@extclass(Softmax)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = SoftmaxOp(child_model)

  shared_children[id(self)] = model 
  return model

@extclass(Relu)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = ReluOp(child_model)
  
  shared_children[id(self)] = model 
  return model

@extclass(Log)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = LogOp(child_model)
  
  shared_children[id(self)] = model 
  return model

@extclass(Exp)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = ExpOp(child_model)
  
  shared_children[id(self)] = model 
  return model

@extclass(Downsample)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = DownsampleOp(child_model, self.in_c, 2, self.name)

  shared_children[id(self)] = model 
  return model

@extclass(Upsample)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = UpsampleOp(child_model, self.in_c, 2, self.name)

  shared_children[id(self)] = model 
  return model

@extclass(Conv1x1)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = Conv1x1Op(child_model, self.in_c, self.out_c, self.groups, self.name)

  shared_children[id(self)] = model 
  return model

@extclass(Conv1D)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = Conv1DOp(child_model, self.in_c, self.out_c, self.groups, self.name, self.kwidth)

  shared_children[id(self)] = model 
  return model


@extclass(Conv2D)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = Conv2DOp(child_model, self.in_c, self.out_c, self.groups, self.name, self.kwidth)

  shared_children[id(self)] = model 
  return model

@extclass(InterleavedSum)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = InterleavedSumOp(child_model, self.out_c)

  shared_children[id(self)] = model 
  return model

@extclass(GroupedSum)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = GroupedSumOp(child_model, self.out_c)

  shared_children[id(self)] = model 
  return model


