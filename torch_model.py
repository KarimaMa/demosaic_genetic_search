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


class ChromaExtractorOp(nn.Module):
  def __init__(self, operand1, operand2, operand3):
    super(ChromaExtractorOp, self).__init__()
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

  def forward(self, chroma_hvq, bayer, green):
    # input: cq, cv, ch and bayer output: red and blue 
    out_shape = list(bayer.shape)
    out_shape[1] = 3
    out = torch.empty(torch.Size(out_shape), device=chroma_hvq.device)

    out[:,0,0::2,0::2] = chroma_hvq[:,0,0::2,0::2]
    out[:,0,1::2,1::2] = chroma_hvq[:,1,1::2,1::2]
    out[:,0,1::2,0::2] = chroma_hvq[:,2,1::2,0::2]
    out[:,0,0::2,1::2] = bayer[:,0,0::2,1::2]

    out[:,1,:,:]       =  green[:,0,:,:]
    out[:,2,1::2,1::2] = chroma_hvq[:,0,1::2,1::2]
    out[:,2,0::2,0::2] = chroma_hvq[:,1,0::2,0::2]
    out[:,2,0::2,1::2] = chroma_hvq[:,2,0::2,1::2]
    out[:,2,1::2,0::2] = bayer[:,0,1::2,0::2]

    return out

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)
    self._operands[2].to_gpu(gpu_id)


class GreenExtractorOp(nn.Module):
  def __init__(self, loperand, roperand, gpu_id=None):
    super(GreenExtractorOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])
    self.mask = torch.zeros((IMG_H, IMG_W))
    self.mask[0::2,1::2] = 1
    self.mask[1::2,0::2] = 1

    # self.chroma_mask = torch.zeros((IMG_H, IMG_W))
    # self.chroma_mask[1::2,1::2] = 1
    # self.chroma_mask[0::2,0::2] = 1

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

  def forward(self, x, bayer):
    # if bayer.is_cuda and not self.mask.is_cuda:
    #   self.mask = self.mask.to(device=f"cuda:{self.gpu_id}")
    _, _, bayer_h, bayer_w = bayer.size()
    #green = x * self.mask[0:bayer_h, 0:bayer_w] + (self.chroma_mask*bayer[:,0,...]).unsqueeze(1)
    green = x * self.mask[0:bayer_h, 0:bayer_w] + bayer[:,0,...].unsqueeze(1)
    return green

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)
    self.mask = self.mask.to(device=f"cuda:{gpu_id}")
    #self.chroma_mask = self.chroma_mask.to(device=f"cuda:{gpu_id}")

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
  def __init__(self, operand, C_in, param_name):
    super(DownsampleOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    pool = nn.Conv2d(C_in, C_in, 5, stride=3, padding=(3,3), bias=False)
    self.param_name = param_name
    setattr(self, param_name, pool)

  def _initialize_parameters(self):
    pool = getattr(self, self.param_name)
    w = 1.0/9.0
    weights = torch.zeros(pool.weight.shape) 
    weights[:,:,0::2,0::2] = w
    pool.weight.data = weights
    pool.weight.requires_grad = False
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

  def forward(self, x):
    pool = getattr(self, self.param_name)
    out = pool(x)
    return out

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class FastUpsampleOp(nn.Module):
  def __init__(self, operand, C_in, param_name):
    super(FastUpsampleOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.in_c = C_in
    self.gpu_id = None
    self.nn_broadcaster = nn.Upsample(scale_factor=3, mode='nearest')
 
  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

  def forward(self, x):
    # upsample each channel separately 
    blu = self.nn_broadcaster(x[:,:,0::2,1::2])
    red = self.nn_broadcaster(x[:,:,1::2,0::2])
    grR = self.nn_broadcaster(x[:,:,1::2,1::2])
    grB = self.nn_broadcaster(x[:,:,0::2,0::2])

    # crop each channel appropriately  
    out_shape = list(x.shape)
    out_shape[2] = (out_shape[2] * 3) - 4
    out_shape[3] = (out_shape[3] * 3) - 4

    out = torch.empty(torch.Size(out_shape), device=x.device)

    out[:,:,0::2,0::2] = grR[:,:,0:-2, 0:-2]
    out[:,:,0::2,1::2] = red[:,:,0:-2, 2: ]
    out[:,:,1::2,0::2] = blu[:,:,2: ,  0:-2]
    out[:,:,1::2,1::2] = grB[:,:,2: ,  2:  ]

    return out 

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self.gpu_id = gpu_id

    
class UpsampleOp(nn.Module):
  def __init__(self, operand, C_in, param_name):
    super(UpsampleOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.in_c = C_in
    # extract predicted values centered at Gr, R, B, Gb into a dense quad
    flat2dense = nn.Conv2d(C_in, C_in*4, 2, padding=0, stride=2, bias=False)
    # splat them into their corresponding locations in an upsampled bayer
    dense2fullres_bayer = nn.ConvTranspose2d(C_in*4, C_in*4, 6, groups=C_in*4, stride=6, bias=False)
    self.param_name_flat2dense = f"{param_name}_flat2dense"
    self.param_name_dense2fullres_bayer = f"{param_name}_dense2fullres_bayer"
    setattr(self, self.param_name_flat2dense, flat2dense)
    setattr(self, self.param_name_dense2fullres_bayer, dense2fullres_bayer)

  def _initialize_parameters(self):
    flat2dense = getattr(self, self.param_name_flat2dense)
    dense2fullres_bayer = getattr(self, self.param_name_dense2fullres_bayer)
    init_flat2dense(flat2dense.weight.data)
    init_dense2fullres_bayer(dense2fullres_bayer.weight.data)
    flat2dense.weight.requires_grad = False
    dense2fullres_bayer.weight.requires_grad = False
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

  def forward(self, x):
    flat2dense = getattr(self, self.param_name_flat2dense)
    dense2fullres_bayer = getattr(self, self.param_name_dense2fullres_bayer)
    # 1st channel centered at Gr, 2nd channel centered at R, 3rd channel at B, 4th channel at Gb
    dense = flat2dense(x)
  
    # spread Gr, R, B, and Gb centered weights
    fullres = dense2fullres_bayer(dense)
    split = torch.split(fullres, self.in_c, dim=1)

    # crop centered values differently
    croppedGr = split[0][:,:,0:-4,0:-4]
    croppedR  = split[1][:,:,0:-4, 4: ]
    croppedB  = split[2][:,:,4: , 0:-4]
    croppedGb = split[3][:,:,4: , 4: ]

    return (croppedGr + croppedR + croppedB + croppedGb)
    
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
    assert C_out % 4 == 0, "Output channels must be divisible by 4 to use separable conv"
    self.param_name_v = f"{param_name}_v"
    self.param_name_h = f"{param_name}_h"
    param_name_diag1 = f"{param_name}_diag1"
    param_name_diag2 = f"{param_name}_diag2"

    v = nn.Conv2d(C_in, C_out//4, (kwidth, 1), bias=False, padding=(kwidth//2, 0))
    h = nn.Conv2d(C_in, C_out//4, (1, kwidth), bias=False, padding=(0, kwidth//2))
    self.diag1 = DiagLRConv(C_in, C_out//4, kwidth, kwidth//2, param_name_diag1)
    self.diag2 = DiagRLConv(C_in, C_out//4, kwidth, kwidth//2, param_name_diag2)

    setattr(self, self.param_name_v, v)
    setattr(self, self.param_name_h, h)

  def _initialize_parameters(self):
    v = getattr(self, self.param_name_v)
    h = getattr(self, self.param_name_h)

    nn.init.xavier_normal_(v.weight)
    nn.init.xavier_normal_(h.weight)
    self.diag1._initialize_parameters()
    self.diag2._initialize_parameters()
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

  def forward(self, x):
    v = getattr(self, self.param_name_v)
    h = getattr(self, self.param_name_h)
    return torch.cat((v(x), h(x), self.diag1(x), self.diag2(x)), 1)

  def to_gpu(self, gpu_id):
    self.diag2.to_gpu(gpu_id)
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


class SumROp(nn.Module):
  def __init__(self, operand):
    super(SumROp, self).__init__()
    self._operands = nn.ModuleList([operand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def reset(self):
    self._operands[0].reset()

  def run(self, model_inputs):
    operand = self._operands[0].run(model_inputs)
    return self.forward(operand)

  def forward(self, x):
    return torch.sum(x, dim=1).unsqueeze(1)  

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

@extclass(ChromaExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  model1 = self.child1.ast_to_model(shared_children)
  model2 = self.child2.ast_to_model(shared_children)
  model3 = self.child3.ast_to_model(shared_children)

  return ChromaExtractorOp(model1, model2, model3)

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
  return DownsampleOp(child_model, self.in_c, self.name)

@extclass(Upsample)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return FastUpsampleOp(child_model, self.in_c, self.name)

@extclass(FastUpsample)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return FastUpsampleOp(child_model, self.in_c, self.name)

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

@extclass(SumR)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  child_model = self.child.ast_to_model(shared_children)
  return SumROp(child_model)
