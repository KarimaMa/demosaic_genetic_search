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



"""
Converts flat activation map to 4 channel-wise groups
of activations
Assumes activation map points are centered on input
bayer values with format 
              Gb B
              R  Gr
Moves the values from the flat activation map into 
channels based on where they're centered with format: 
             Gr R B Gb
If input flat activation map has k channels then the 
output will have 4*k channels
"""
def init_flat2dense(weights):
  assert(weights.shape[0] == weights.shape[1]*4), \
  "Expected output channels of flat2dense to be 4x the input channels"
  assert(weights.shape[-2:] == (2,2)), "Expected kernel size of flat2dense to be (2,2)"
  weights.fill_(0.0)
  c_stride = weights.shape[1]
  offset = 0
  for i in range(c_stride):
    weights[offset+i, i, 1, 1] = 1

  offset += c_stride
  for i in range(c_stride):
    weights[offset+i, i, 1, 0] = 1

  offset += c_stride
  for i in range(c_stride):
    weights[offset+i, i, 0, 1] = 1

  offset += c_stride
  for i in range(c_stride):
    weights[offset+i, i, 0, 0] = 1

"""
splats values of a 4 group channel-wise activation map centered 
at Gr, R, B, Gb in a downsampled input bayer to their locations in an upsampled 
4 group channel-wise bayer. 
Assumes that the input channel order is Gr, R, B, Gb
"""
def init_dense2fullres_bayer(weights):
  assert(weights.shape[-2:] == (6, 6)), "Expected kernel size of dense2fullres_bayer to be (6,6)"
  weights.fill_(0.0)
  c_stride = weights.shape[0]//4
  weights[0         :c_stride  , 0, 0::2, 0::2] = 1
  weights[c_stride  :c_stride*2, 0, 0::2, 1::2] = 1
  weights[c_stride*2:c_stride*3, 0, 1::2, 0::2] = 1
  weights[c_stride*3:          , 0, 1::2, 1::2] = 1


class InputOp(nn.Module):
  def __init__(self, input_name, model=None, model_name=None):
    super(InputOp, self).__init__()
    self.name = input_name
    if model:
      self.model = model
      self.model_name = f"input_model_{model_name}"
      setattr(self, self.model_name, model)
      self.output = None

  def _initialize_parameters(self):
    if hasattr(self, "model"):
      self.model._initialize_parameters()

  def reset(self):
    self.output = None

  # unnecessary function, never called but needed by nn.Module
  def foward(self, model_inputs):
    return model_inputs[self.name]

  def run(self, model_inputs):
    if hasattr(self, "model"):
      if not self.output is None:
        return self.output
      else:
        self.output = self.model.run(model_inputs)
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

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()

  def run(self, model_inputs):
    loperand = self._operands[0].run(model_inputs)
    roperand = self._operands[1].run(model_inputs)
    return self.forward(loperand, roperand)

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

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()

  def run(self, model_inputs):
    loperand = self._operands[0].run(model_inputs)
    roperand = self._operands[1].run(model_inputs)
    return self.forward(loperand, roperand)

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

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()

  def run(self, model_inputs):
    loperand = self._operands[0].run(model_inputs)
    roperand = self._operands[1].run(model_inputs)
    return self.forward(loperand, roperand)

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

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()

  def run(self, model_inputs):
    loperand = self._operands[0].run(model_inputs)
    roperand = self._operands[1].run(model_inputs)
    return self.forward(loperand, roperand)

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

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()

  def run(self, model_inputs):
    loperand = self._operands[0].run(model_inputs)
    roperand = self._operands[1].run(model_inputs)
    return self.forward(loperand, roperand)

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

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()

  def run(self, model_inputs):
    loperand = self._operands[0].run(model_inputs)
    roperand = self._operands[1].run(model_inputs)
    return self.forward(loperand, roperand)

  def forward(self, x, y):
    return torch.cat((x, y), 1)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)


# Takes BayerQuad, 4 channel green prediction, and 6 channel Chroma prediction
# spits out full RGB image at full resolution 
class RGBExtractorOp(nn.Module):
  def __init__(self, operand1, operand2, operand3):
    super(RGBExtractorOp, self).__init__()
    self._operands = nn.ModuleList([operand1, operand2, operand3])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()
    self._operands[2]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()
    self._operands[2].reset()

  def run(self, model_inputs):
    self.reset()
    operand1 = self._operands[0].run(model_inputs)
    operand2 = self._operands[1].run(model_inputs)
    operand3 = self._operands[2].run(model_inputs)

    return self.forward(operand1, operand2, operand3)

  def forward(self, bayer_quad, green_pred, chroma_pred):
    # bayer_quad : 4 channels
    # green_pred : 4 channels Green 
    # chroma_pred: 6 channels Red at Gr, B, Gb and Blue at Gr, R, Gb
    mosaic_shape = list(bayer_quad.shape)
    out_shape = [mosaic_shape[0], 3, mosaic_shape[2]*2, mosaic_shape[3]*2]
    img = torch.empty(torch.Size(out_shape), device=bayer_quad.device)

    img[:,0,0::2,0::2] = chroma_pred[:,0,:,:]
    img[:,0,0::2,1::2] = bayer_quad[:,1,:,:]
    img[:,0,1::2,0::2] = chroma_pred[:,1,:,:]
    img[:,0,1::2,1::2] = chroma_pred[:,2,:,:]

    img[:,1,0::2,0::2] = bayer_quad[:,0,:,:]
    img[:,1,0::2,1::2] = green_pred[:,1,:,:]
    img[:,1,1::2,0::2] = green_pred[:,2,:,:]
    img[:,1,1::2,1::2] = bayer_quad[:,3,:,:]

    img[:,2,0::2,0::2] = chroma_pred[:,3,:,:]
    img[:,2,0::2,1::2] = chroma_pred[:,4,:,:]
    img[:,2,1::2,0::2] = bayer_quad[:,2,:,:]
    img[:,2,1::2,1::2] = chroma_pred[:,5,:,:]
    
    return img 

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)
    self._operands[2].to_gpu(gpu_id)


class GreenExtractorOp(nn.Module):
  def __init__(self, operand1, operand2):
    super(GreenExtractorOp, self).__init__()
    self._operands = nn.ModuleList([operand1, operand2])
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()
    self._operands[1].reset()

  def run(self, model_inputs):
    self.reset()
    operand1 = self._operands[0].run(model_inputs)
    operand2 = self._operands[1].run(model_inputs)

    return self.forward(operand1, operand2)

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


class SoftmaxOp(nn.Module):
  def __init__(self, operand):
    super(SoftmaxOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.f = nn.Softmax(dim=1)
  
  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

  def forward(self, x):
    return self.f(x)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class ReluOp(nn.Module):
  def __init__(self, operand):
    super(ReluOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.f = nn.ReLU()

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

  def forward(self, x):
    return self.f(x)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class LogOp(nn.Module):
  def __init__(self, operand):
    super(LogOp, self).__init__()
    self._operands = nn.ModuleList([operand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

  def forward(self, x):
    return torch.log(x)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class ExpOp(nn.Module):
  def __init__(self, operand):
    super(ExpOp, self).__init__()
    self._operands = nn.ModuleList([operand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

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

  def _initialize_parameters(self):
    downsampler = getattr(self, self.param_name)
    nn.init.xavier_normal_(downsampler.weight)
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

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

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

  def forward(self, x):
    upsampler = getattr(self, self.param_name)
    return upsampler(x)
    
  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class Conv1x1Op(nn.Module):
  def __init__(self, operand, C_in, C_out, param_name):
    super(Conv1x1Op, self).__init__()
    self._operands = nn.ModuleList([operand])
    f = nn.Conv2d(C_in, C_out, (1, 1), bias=False, padding=0)
    self.param_name = param_name
    setattr(self, param_name, f)

  def _initialize_parameters(self):
    f = getattr(self, self.param_name)
    nn.init.xavier_normal_(f.weight)
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

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
  def __init__(self, operand, C_in, C_out, param_name, kwidth):
    super(Conv1DOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    assert C_out % 2 == 0, "Output channels must be divisible by 2 to use separable conv"
    self.param_name_v = f"{param_name}_v"
    self.param_name_h = f"{param_name}_h"

    v = nn.Conv2d(C_in, C_out//4, (kwidth, 1), bias=False, padding=(kwidth//2, 0))
    h = nn.Conv2d(C_in, C_out//4, (1, kwidth), bias=False, padding=(0, kwidth//2))
  
    setattr(self, self.param_name_v, v)
    setattr(self, self.param_name_h, h)

  def _initialize_parameters(self):
    v = getattr(self, self.param_name_v)
    h = getattr(self, self.param_name_h)

    nn.init.xavier_normal_(v.weight)
    nn.init.xavier_normal_(h.weight)   
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

  def forward(self, x):
    v = getattr(self, self.param_name_v)
    h = getattr(self, self.param_name_h)
    return torch.cat((v(x), h(x)), 1)

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class Conv2DOp(nn.Module):
  def __init__(self, operand, C_in, C_out, param_name, kwidth):
    super(Conv2DOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.param_name = param_name
    f = nn.Conv2d(C_in, C_out, (kwidth,kwidth), bias=False, padding=kwidth//2)
    setattr(self, self.param_name, f)

  def _initialize_parameters(self):
    f = getattr(self, self.param_name)
    nn.init.xavier_normal_(f.weight)
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

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

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

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

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

  def forward(self, x):
    x_shape = list(x.shape)
    x_reshaped = torch.reshape(x, (x_shape[0], x_shape[1]//self.C_out, self.C_out, x_shape[2], x_shape[3]))
    out = x_reshaped.sum(1, keepdim=False) 
    return out
    
  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


@extclass(Input)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if hasattr(self, "node"):
    if id(self) in shared_children:
      return shared_children[id(self)]
    else:
      node_model = self.node.ast_to_model()
      input_op = InputOp(self.name, model=node_model, model_name=self.name)
      shared_children[id(self)] = input_op
    return input_op
  return InputOp(self.name)

@extclass(Add)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  return AddOp(lmodel, rmodel)

@extclass(Sub)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  return SubOp(lmodel, rmodel)

@extclass(Mul)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  return MulOp(lmodel, rmodel)

@extclass(LogSub)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  lmodel = self.lchild.ast_to_model(shared_children)
  if not type(self.rchild.parent) is tuple:
    print(f"right child of logsub parents {self.rchild.parent}")
    print(self.rchild.parent.dump())
    print(self.rchild.dump())
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
  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  return StackOp(lmodel, rmodel)

@extclass(RGBExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  model1 = self.child1.ast_to_model(shared_children)
  model2 = self.child2.ast_to_model(shared_children)
  model3 = self.child3.ast_to_model(shared_children)
  return RGBExtractorOp(model1, model2, model3)

@extclass(GreenExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  return GreenExtractorOp(lmodel, rmodel)

@extclass(Softmax)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return SoftmaxOp(child_model)

@extclass(Relu)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return ReluOp(child_model)

@extclass(Log)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return LogOp(child_model)

@extclass(Exp)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return ExpOp(child_model)

@extclass(Downsample)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return DownsampleOp(child_model, self.in_c, 2, self.name)

@extclass(Upsample)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return UpsampleOp(child_model, self.in_c, 2, self.name)

@extclass(Conv1x1)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return Conv1x1Op(child_model, self.in_c, self.out_c, self.name)

@extclass(Conv1D)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return Conv1DOp(child_model, self.in_c, self.out_c, self.name, self.kwidth)

@extclass(Conv2D)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return Conv2DOp(child_model, self.in_c, self.out_c, self.name, self.kwidth)

@extclass(InterleavedSum)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return InterleavedSumOp(child_model, self.out_c)

@extclass(GroupedSum)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return GroupedSumOp(child_model, self.out_c)

