"""
      ieturn out
lowered representation of AST to pytorch model
"""

import numpy
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
        # print(f"loading pretrained weights for input model {self.name} no grad {self.no_grad}")
        state_dict = torch.load(self.weights, map_location="cpu")
        self.model.load_state_dict(state_dict)
      else:
        # print(f"initializing weight for input model {self.name} no grad {self.no_grad}")
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


class SRGBExtractorOp(nn.Module):
  def __init__(self, green, chromapred):
    super(SRGBExtractorOp, self).__init__()
    self._operands = nn.ModuleList([green, chromapred])
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

  def forward(self, green, chromapred):
    out_shape = list(chromapred.shape)
    out_shape[1] = 3
    img = torch.empty(torch.Size(out_shape), device=chromapred.device)

    # fill in reds
    img[:,0,:,:] = chromapred[:,0,:,:]
    img[:,1,:,:] = green[:,0,:,:]
    img[:,2,:,:] = chromapred[:,1,:,:]

    return img 

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


"""
Inputs are all on flat xtrans resolution = image resolution
  green prediction: 1 channels
  xtrans: 3 channels
  chroma predctions: 2 channels
"""
class XFlatRGBExtractorOp(nn.Module):
  def __init__(self, green_pred, xtrans, chroma_pred):
    super(XFlatRGBExtractorOp, self).__init__()
    self._operands = nn.ModuleList([green_pred, xtrans, chroma_pred])
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor=6)
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

  def forward(self, green_pred, xtrans, chroma_pred):
    out_shape = list(xtrans.shape)
    out_shape[1] = 3

    img = torch.empty(torch.Size(out_shape), device=xtrans.device)
    img[:,1,:,:] = green_pred[:,0,:,:]
    img[:,0,:,:] = xtrans[:,0,:,:]
    img[:,2,:,:] = xtrans[:,2,:,:]

    factor = 6

    g_pos = [(0,0),        (0,2), (0,3),        (0,5),
                  (1,1),               (1,4),
           (2,0),        (2,2), (2,3),        (2,5),
           (3,0),        (3,2), (3,3),        (3,5),
                  (4,1),               (4,4),
           (5,0),        (5,2), (5,3),        (5,5)]
    r_pos = [(0,4),
             (1,0), (1,2),
             (2,4),
             (3,1),
             (4,3), (4,5),
             (5,1)]
    b_pos = [(0,1),
             (1,3), (1,5),
             (2,1),
             (3,4),
             (4,0), (4,2),
             (5,4)]

    # chroma at green
    for i, pos in enumerate(g_pos):
      img[:,0,pos[0]::factor, pos[1]::factor] = chroma_pred[:, 0, pos[0]::factor, pos[1]::factor]
      img[:,2,pos[0]::factor, pos[1]::factor] = chroma_pred[:, 1, pos[0]::factor, pos[1]::factor]

    for i, pos in enumerate(b_pos):
      img[:,0,pos[0]::factor, pos[1]::factor] = chroma_pred[:, 0, pos[0]::factor, pos[1]::factor]
    for i, pos in enumerate(r_pos):
      img[:,2,pos[0]::factor, pos[1]::factor] = chroma_pred[:, 1, pos[0]::factor, pos[1]::factor]

    return img

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)
    self._operands[2].to_gpu(gpu_id)


class XRGBExtractorOp(nn.Module):
  def __init__(self, green_pred, xtrans, chroma_pred):
    super(XRGBExtractorOp, self).__init__()
    self._operands = nn.ModuleList([green_pred, xtrans, chroma_pred])
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor=6)
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

  def forward(self, green_pred, xtrans, chroma_pred):
    # green_pred : 36 channels
    # rb_xtrans : 16 channels  
    # chroma_pred: 56 channels red at greens, red at blues, blue at reds, blue at greens
    packed_out_shape = list(green_pred.shape)
    packed_out_shape[1] = 108

    packed_img = torch.empty(torch.Size(packed_out_shape), device=green_pred.device)

    g_block1_pos = [(0,0),        (0,2), 
                          (1,1),             
                   (2,0),        (2,2)]
    g_block2_pos = [(0,3),        (0,5),
                          (1,4),
                   (2,3),        (2,5)]
    g_block3_pos = [(3,0),        (3,2),
                          (4,1),            
                   (5,0),        (5,2)]
    g_block4_pos = [(3,3),        (3,5),
                          (4,4),
                   (5,3),        (5,5)]
    g_pos = g_block1_pos + g_block2_pos + g_block3_pos + g_block4_pos

    r_block1_pos = [(1,0), (1,2)]
    r_block2_pos = [(0,4), (2,4)]
    r_block3_pos = [(3,1), (5,1)]
    r_block4_pos = [(4,3), (4,5)]

    r_pos = r_block1_pos + r_block2_pos + r_block3_pos + r_block4_pos

    b_block1_pos = [(0,1), (2,1)]
    b_block2_pos = [(1,3), (1,5)]
    b_block3_pos = [(4,0), (4,2)]
    b_block4_pos = [(3,4), (5,4)]
 
    b_pos = b_block1_pos + b_block2_pos + b_block3_pos + b_block4_pos

    r_at_b_pred_pos = [0,3,5,6,9,10,12,15]
    b_at_r_pred_pos = [1,2,4,7,8,11,13,14]

    out_offset = 0
    in_offset = 0
    # red at green
    for i, pos in enumerate(g_pos):
      out_c = pos[0]*6 + pos[1] + out_offset
      in_c = in_offset + i
      packed_img[:,out_c,:,:] = chroma_pred[:,in_c,:,:]

    # red at blue
    in_offset += len(g_pos)
    for i, pos in enumerate(b_pos):
      out_c = pos[0]*6 + pos[1] + out_offset
      in_c = in_offset + r_at_b_pred_pos[i]
      packed_img[:,out_c,:,:] = chromapred[:,in_c,:,:]

    # red from xtrans
    for i, pos in enumerate(r_pos):
      out_c = pos[0]*6 + pos[1] + out_offset
      in_c = pos[0]*6 + pos[1]
      packed_img[:,out_c,:,:] = xtrans[:,in_c,:,:]

    # green 
    out_offset += 36
    packed_img[:,out_offset:out_offset+36,:,:] = green_pred[:,:,:,:]

    # blue at red
    out_offset += 36
    for i, pos in enumerate(r_pos):
      out_c = pos[0]*6 + pos[1] + out_offset
      in_c = in_offset + b_at_r_pred_pos[i]
      packed_img[:,out_c,:,:] = chromapred[:,in_c,:,:]

    # blue at green
    in_offset += (len(r_pos) + len(b_pos))
    for i, pos in enumerate(g_pos):
      out_c = pos[0]*6 + pos[1] + out_offset
      in_c = in_offset + i
      packed_img[:,out_c,:,:] = chroma_pred[:,in_c,:,:]

    # blue from xtrans
    for i, pos in enumerate(b_pos):
      out_c = pos[0]*6 + pos[1] + out_offset
      in_c = pos[0]*6 + pos[1]
      packed_img[:,out_c,:,:] = xtrans[:,in_c,:,:]

    img = self.pixel_shuffle(packed_img)
    
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
  Takes bayer quad and 8 channel prediction
  Spits out full rgb  
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


class FlatRGB8ChanExtractorOp(nn.Module):
  def __init__(self, operand1, operand2):
    super(FlatRGB8ChanExtractorOp, self).__init__()
    self._operands = nn.ModuleList([operand1, operand2])
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
  Takes 3chan bayer and 8 channel prediction
  Spits out full rgb  
  """
  def forward(self, bayer3chan, rgb_pred):
    # input: bayer an rgb missing values
    # output: full image
    out_shape = list(bayer3chan.shape)
    output = torch.empty(torch.Size(out_shape), device=bayer3chan.device)
    
    output[:,:,:,:] = bayer3chan[:,:,:,:]

    output[:,0,:,:] = rgb_pred[:,0,:,:]
    output[:,1,:,:] = rgb_pred[:,1,:,:]
    output[:,2,:,:] = rgb_pred[:,2,:,:]

    output[:,0,0::2,1::2] = bayer3chan[:,0,0::2,1::2]
    output[:,1,0::2,0::2] = bayer3chan[:,1,0::2,0::2]
    output[:,1,1::2,1::2] = bayer3chan[:,1,1::2,1::2]
    output[:,2,1::2,0::2] = bayer3chan[:,2,1::2,0::2]

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


class SGreenExtractorOp(nn.Module):
  def __init__(self, operand):
    super(SGreenExtractorOp, self).__init__()
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

  def forward(self, green_pred):
    return green_pred

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class XGreenExtractorOp(nn.Module):
  def __init__(self, operand1, operand2):
    super(XGreenExtractorOp, self).__init__()
    self._operands = nn.ModuleList([operand1, operand2])
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor=6)
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
  Takes Xtrans mosaic and 16 channel green prediction
  Spits out full green channel 
  """
  def forward(self, xtrans, green_pred):
    out = self.pixel_shuffle(xtrans)
    # fill in red locations
    out[:,0,0::6,4::6] = green_pred[:,0,:,:]
    
    out[:,0,1::6,0::6] = green_pred[:,1,:,:]
    out[:,0,1::6,2::6] = green_pred[:,2,:,:]
    
    out[:,0,2::6,4::6] = green_pred[:,3,:,:]
    
    out[:,0,3::6,1::6] = green_pred[:,4,:,:]
    
    out[:,0,4::6,3::6] = green_pred[:,5,:,:]
    out[:,0,4::6,5::6] = green_pred[:,6,:,:]

    out[:,0,5::6,1::6] = green_pred[:,7,:,:]

    # fill in blue locations
    out[:,0,0::6,1::6] = green_pred[:,8,:,:]
    
    out[:,0,1::6,3::6] = green_pred[:,9,:,:]
    out[:,0,1::6,5::6] = green_pred[:,10,:,:]
    
    out[:,0,2::6,1::6] = green_pred[:,11,:,:]
    
    out[:,0,3::6,4::6] = green_pred[:,12,:,:]
    
    out[:,0,4::6,0::6] = green_pred[:,13,:,:]
    out[:,0,4::6,2::6] = green_pred[:,14,:,:]
    
    out[:,0,5::6,4::6] = green_pred[:,15,:,:]

    return out

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)
    self._operands[1].to_gpu(gpu_id)



class XFlatGreenExtractorOp(nn.Module):
  def __init__(self, operand1, operand2):
    super(XFlatGreenExtractorOp, self).__init__()
    self._operands = nn.ModuleList([operand1, operand2])
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor=6)
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
  Takes 3 channel flat Xtrans mosaic and 1 channel green prediction
  Spits out full green channel 
  """
  def forward(self, xtrans, green_pred):
    outsize = list(xtrans.shape)
    outsize[1] = 1
    out = torch.empty(torch.Size(outsize), device=xtrans.device)
    out[:,0,:,:] = xtrans[:,1,:,:]

    # fill in red locations
    out[:,0,0::6,4::6] = green_pred[:,0,0::6,4::6]
    
    out[:,0,1::6,0::6] = green_pred[:,0,1::6,0::6]
    out[:,0,1::6,2::6] = green_pred[:,0,1::6,2::6]
    
    out[:,0,2::6,4::6] = green_pred[:,0,2::6,4::6]
    
    out[:,0,3::6,1::6] = green_pred[:,0,3::6,1::6]
    
    out[:,0,4::6,3::6] = green_pred[:,0,4::6,3::6]
    out[:,0,4::6,5::6] = green_pred[:,0,4::6,5::6]

    out[:,0,5::6,1::6] = green_pred[:,0,5::6,1::6]

    # fill in blue locations
    out[:,0,0::6,1::6] = green_pred[:,0,0::6,1::6]
    
    out[:,0,1::6,3::6] = green_pred[:,0,1::6,3::6]
    out[:,0,1::6,5::6] = green_pred[:,0,1::6,5::6]
    
    out[:,0,2::6,1::6] = green_pred[:,0,2::6,1::6]

    out[:,0,3::6,4::6] = green_pred[:,0,3::6,4::6]
    
    out[:,0,4::6,0::6] = green_pred[:,0,4::6,0::6]
    out[:,0,4::6,2::6] = green_pred[:,0,4::6,2::6]

    out[:,0,5::6,4::6] = green_pred[:,0,5::6,4::6]

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
takes flat 1 channel green prediction and extracts out green predicted 
values at Red and Blue Xtrans locations in 3x3 grid row major order
returning values in packed 6x6 resolution
"""
class XGreenRBExtractorOp(nn.Module):
  def __init__(self, operand):
    super(XGreenRBExtractorOp, self).__init__()
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
    factor = 6

    out_shape = list(flat_green.shape)
    out_shape[1] = 16
    out_shape[2] //= factor
    out_shape[3] //= factor

    num_blocks = 4
    blocks_x = 2
    blocks_y = 2

    block_w = 3
    block_h = 3
    block_size = 4 # 4 red and blue locations per 3x3 block
    
    green_rb = torch.empty(torch.Size(out_shape), device=flat_green.device)

    for block in range(num_blocks):
      for i in range(block_size):
        c = block * block_size + i 
        # coordinate within the 6x6 group of 2x2 blocks 
        x = (block % blocks_y) * block_w + (i*2+1) % block_w
        y = (block // blocks_x) * block_h + (i*2+1) // block_w
        green_rb[:,c,:,:] = flat_green[:,0, y::factor, x::factor] 

    return green_rb

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


class LearnedDownsampleOp(nn.Module):
  def __init__(self, operand, C_in, C_out, scale_factor, groups, param_name):
    super(LearnedDownsampleOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.scale_factor = scale_factor
    self.downsample_w = scale_factor * 2 + (scale_factor % 2) # use odd kernel width for odd sampling factors 
    downsampler = nn.Conv2d(C_in, C_out, self.downsample_w, stride=scale_factor, groups=groups, padding=math.ceil((self.downsample_w-self.scale_factor)/2), bias=False)
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
    out = downsampler(x)
    if out.shape[2] != x.shape[2] / self.scale_factor:
      print(f'down input size {x.shape}')
      print(f'output shape {out.shape}')
    return out

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


"""
performs a unique convolution per spatial location within a period
"""
class PeriodicConvOp(nn.Module):
  def __init__(self, operand, C_in, C_out, period, kwidth, param_name):
    super(PeriodicConvOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.period = period
    self.kwidth = kwidth
    self.param_name = param_name

    im2col = nn.Unfold(kwidth, padding=kwidth//2)

    input_c = (period**2 * C_in * kwidth**2)
    output_c = (period**2 * C_out)
    conv = nn.Conv2d(input_c, output_c, 1, groups=self.period**2, padding=0, bias=False)
    unpack = nn.PixelShuffle(upscale_factor=period)
    
    setattr(self, param_name+"_im2col", im2col)
    setattr(self, param_name+"_conv", conv)
    setattr(self, param_name+"_unpack", unpack)

    self.output = None

  def _initialize_parameters(self):
    conv = getattr(self, self.param_name+"_conv")
    nn.init.xavier_normal_(conv.weight)
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
    im2col = getattr(self, self.param_name+"_im2col")
    conv = getattr(self, self.param_name+"_conv")
    unpack = getattr(self, self.param_name+"_unpack")

    tiles = im2col(x) # batch_size, c * kwidth**2, h * w
    # reshape back to h, w dimensions
    tiled = torch.reshape(tiles, (-1, (x.shape[1]*self.kwidth**2), x.shape[2], x.shape[3]))

    # pack image by period
    packed_size = list(tiled.shape) 
    packed_size[1] *= self.period**2
    packed_size[2] //= self.period
    packed_size[3] //= self.period
    packed = torch.empty(torch.Size(packed_size), device=x.device) # batch_size, (period**2 * c_in * kwidth**2), h//period, w//period

    period_size = self.period**2
    channels_per_conv = tiled.shape[1]
    for i in range(period_size):
      outc = i * channels_per_conv 
      x_loc = i % self.period
      y_loc = i // self.period
      packed[:,outc:outc+channels_per_conv,:,:] = tiled[:, :, y_loc::self.period, x_loc::self.period] 
    
    conv_output = conv(packed) # batch_size, (c_out * period**2), h//period, w//period
    out = unpack(conv_output) # batch_size, c_out, h, w
    return out

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class PeriodicConvV2Op(nn.Module):
  def __init__(self, operand, C_in, C_out, period, kwidth, param_name):
    super(PeriodicConvV2Op, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.period = period
    self.kwidth = kwidth
    self.param_name = param_name
    self.out_c = C_out 
    conv = nn.Conv2d(C_in, C_out*period**2, kwidth, padding=kwidth//2, bias=False)
    setattr(self, param_name, conv)

    self.output = None

  def _initialize_parameters(self):
    conv = getattr(self, self.param_name)
    nn.init.xavier_normal_(conv.weight)
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
   
    conv = getattr(self, self.param_name)
    conv_out = conv(x)

    out_size = list(x.shape) 
    out_size[1] = self.out_c 
    out = torch.empty(torch.Size(out_size), device=x.device) 

    period_size = self.period**2

    period_size = self.period**2
    for p in range(period_size):
      x = p % self.period
      y = p // self.period 
      out[:, :, y::self.period, x::self.period] = conv_out[:, p:p+self.out_c, y::self.period, x::self.period]

    return out

  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class PackOp(nn.Module):
  def __init__(self, operand, factor):
    super(PackOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.scale_factor = factor
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
    factor = self.scale_factor
    packed_size = list(x.shape)
    packed_size[1] *= factor**2
    packed_size[2] //= factor
    packed_size[3] //= factor
    packed = torch.empty(torch.Size(packed_size), device=x.device)

    for c in range(x.shape[1]):
      for i in range(factor):
        for j in range(factor):
          outc = c * factor**2 + i * factor + j
          packed[:,outc,:,:] = x[:, c, i::factor, j::factor] 
    return packed
    
  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class LearnedPackOp(nn.Module):
  def __init__(self, operand, C_in, C_out, factor, param_name):
    super(LearnedPackOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.scale_factor = factor
    f = nn.Conv2d(C_in, C_out, factor, stride=factor, bias=False, padding=0)
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


"""
Unpacks the mosaic
"""
class UnpackOp(nn.Module):
  def __init__(self, operand, C_in, factor):
    super(UnpackOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.in_c = C_in
    self.scale_factor = factor
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor=factor)
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
    return self.pixel_shuffle(x)
    
  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class BilinearUpsampleOp(nn.Module):
  def __init__(self, operand, C_in, scale_factor, param_name):
    super(BilinearUpsampleOp, self).__init__()
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
    out = upsampler(x)
    if out.shape[2] != x.shape[2] * self.scale_factor:
      print(f'up input shape {x.shape}')
      print(f'out shape {out.shape}')
    return out
    
  def to_gpu(self, gpu_id):
    self._operands[0].to_gpu(gpu_id)


class LearnedUpsampleOp(nn.Module):
  def __init__(self, operand, C_in, C_out, scale_factor, groups, param_name):
    super(LearnedUpsampleOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.in_c = C_in
    self.out_c = C_out
    self.scale_factor = scale_factor
    self.param_name = param_name
    upsampler = nn.ConvTranspose2d(self.in_c, self.out_c, scale_factor, groups=groups, stride=scale_factor)

    setattr(self, self.param_name, upsampler)
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

    kwargs = {"model":node_model, "model_name":self.name, "no_grad":self.no_grad}

    if hasattr(self, "weight_file"):
      kwargs["weights"] = self.weight_file
    input_op = InputOp(self.name, **kwargs)
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
  self.model = model
  
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
  self.model = model
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
  self.model = model
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
  model = LogSubOp(lmodel, rmodel)
  self.model = model
  return model

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
  model = AddExpOp(lmodel, rmodel)
  self.model = model
  return model

@extclass(Stack)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  model = StackOp(lmodel, rmodel)
  self.model = model
  
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
  self.model = model
  
  shared_children[id(self)] = model 
  return model

@extclass(XFlatRGBExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child1_model = self.child1.ast_to_model(shared_children)
  child2_model = self.child2.ast_to_model(shared_children)
  child3_model = self.child3.ast_to_model(shared_children)
  model = XFlatRGBExtractorOp(child1_model, child2_model, child3_model)

  self.model = model
  shared_children[id(self)] = model 
  return model

@extclass(XRGBExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child1_model = self.child1.ast_to_model(shared_children)
  child2_model = self.child2.ast_to_model(shared_children)
  child3_model = self.child3.ast_to_model(shared_children)
  model = XRGBExtractorOp(child1_model, child2_model, child3_model)

  self.model = model
  shared_children[id(self)] = model 
  return model


@extclass(SRGBExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child1_model = self.lchild.ast_to_model(shared_children)
  child2_model = self.rchild.ast_to_model(shared_children)
  model = SRGBExtractorOp(child1_model, child2_model)

  self.model = model
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
  self.model = model

  shared_children[id(self)] = model 
  return model

@extclass(FlatRGB8ChanExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  model = FlatRGB8ChanExtractorOp(lmodel, rmodel)
  
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
  self.model = model
  
  shared_children[id(self)] = model 
  return model

@extclass(SGreenExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child = self.child.ast_to_model(shared_children)
  model = SGreenExtractorOp(child)
  
  shared_children[id(self)] = model 
  return model

@extclass(XGreenExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  model = XGreenExtractorOp(lmodel, rmodel)
  self.model = model
    
  shared_children[id(self)] = model 
  return model

@extclass(XFlatGreenExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  lmodel = self.lchild.ast_to_model(shared_children)
  rmodel = self.rchild.ast_to_model(shared_children)
  model = XFlatGreenExtractorOp(lmodel, rmodel)
  self.model = model
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
  self.model = model
  
  shared_children[id(self)] = model 
  return model

@extclass(XGreenRBExtractor)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = XGreenRBExtractorOp(child_model)
  self.model = model
  
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
  self.model = model
  
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
  self.model = model
  
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
  self.model = model
  
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
  self.model = model
  
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
  self.model = model
  
  shared_children[id(self)] = model 
  return model

@extclass(LearnedDownsample)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = LearnedDownsampleOp(child_model, self.in_c, self.out_c, int(self.factor), self.groups, self.name)
  self.model = model
  
  shared_children[id(self)] = model 
  return model

# @extclass(PeriodicConv)
# def ast_to_model(self, shared_children=None):
#   if shared_children is None:
#     shared_children = {}
#   if id(self) in shared_children:
#     return shared_children[id(self)]

#   child_model = self.child.ast_to_model(shared_children)
#   model = PeriodicConvV2Op(child_model, self.in_c, self.out_c, self.period, self.kwidth, self.name)

#   shared_children[id(self)] = model 
#   return model

@extclass(Pack)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = PackOp(child_model, int(self.factor))
  self.model = model
  
  shared_children[id(self)] = model 
  return model

# @extclass(LearnedPack)
# def ast_to_model(self, shared_children=None):
#   if shared_children is None:
#     shared_children = {}
#   if id(self) in shared_children:
#     return shared_children[id(self)]

#   child_model = self.child.ast_to_model(shared_children)
#   model = LearnedPackOp(child_model, self.in_c, self.out_c, self.factor, self.name)

#   shared_children[id(self)] = model 
#   return model


@extclass(Unpack)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = UnpackOp(child_model, self.in_c, int(self.factor))
  self.model = model
  
  shared_children[id(self)] = model 
  return model

@extclass(BilinearUpsample)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = BilinearUpsampleOp(child_model, self.in_c, int(self.factor), self.name)
  self.model = model
  
  shared_children[id(self)] = model 
  return model

@extclass(LearnedUpsample)
def ast_to_model(self, shared_children=None):
  if shared_children is None:
    shared_children = {}
  if id(self) in shared_children:
    return shared_children[id(self)]

  child_model = self.child.ast_to_model(shared_children)
  model = LearnedUpsampleOp(child_model, self.in_c, self.out_c, int(self.factor), self.groups, self.name)
  self.model = model
  
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
  self.model = model
  
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
  self.model = model
  
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
  self.model = model
  
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
  self.model = model
  
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
  self.model = model
  
  shared_children[id(self)] = model 
  return model


