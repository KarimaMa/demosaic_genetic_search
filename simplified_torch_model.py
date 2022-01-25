import math
import torch 
import torch.nn as nn



class _AddOp(nn.Module):
  def __init__(self):
    super(_AddOp, self).__init__()

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


class _SubOp(nn.Module):
  def __init__(self):
    super(_SubOp, self).__init__()

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


class _MulOp(nn.Module):
  def __init__(self):
    super(_MulOp, self).__init__()

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


class _StackOp(nn.Module):
  def __init__(self):
    super(_StackOp, self).__init__()
  
  def forward(self, x, y):
    return torch.cat((x, y), 1)


class _SRGBExtractorOp(nn.Module):
  def __init__(self):
    super(_SRGBExtractorOp, self).__init__()

  def forward(self, green, chromapred):
    out_shape = list(chromapred.shape)
    out_shape[1] = 3
    img = torch.empty(torch.Size(out_shape), device=chromapred.device)

    # fill in reds
    img[:,0,:,:] = chromapred[:,0,:,:]
    img[:,1,:,:] = green[:,0,:,:]
    img[:,2,:,:] = chromapred[:,1,:,:]

    return img 


class _SExtractorOp(nn.Module):
  def __init__(self):
    super(_SExtractorOp, self).__init__()

  def forward(self, pred):
    return pred


# Takes FullGreenQuad predction, RedBlueQuad from Bayer, and 
# 6 channel chroma quad prection: R@Gr, B@R, R@B, R@Gb, B@Gr, B@Gb
# spits out full RGB Image at full resolution
class _RGBExtractorOp(nn.Module):
  def __init__(self):
    super(_RGBExtractorOp, self).__init__()

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
class _XFlatRGBExtractorOp(nn.Module):
  def __init__(self):
    super(_XFlatRGBExtractorOp, self).__init__()

  #def forward(self, green_pred, xtrans, chroma_pred):
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


class _XRGBExtractorOp(nn.Module):
  def __init__(self):
    super(_XRGBExtractorOp, self).__init__()

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


class _RGB8ChanExtractorOp(nn.Module):
  def __init__(self):
    super(_RGB8ChanExtractorOp, self).__init__()

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


class _FlatRGB8ChanExtractorOp(nn.Module):
  def __init__(self):
    super(_FlatRGB8ChanExtractorOp, self).__init__()

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


class _GreenExtractorOp(nn.Module):
  def __init__(self):
    super(_GreenExtractorOp, self).__init__()

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


class _SGreenExtractorOp(nn.Module):
  def __init__(self):
    super(_SGreenExtractorOp, self).__init__()

  def forward(self, green_pred):
    return green_pred


class _XGreenExtractorOp(nn.Module):
  def __init__(self):
    super(_XGreenExtractorOp, self).__init__()

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

 
class _XFlatGreenExtractorOp(nn.Module):
  def __init__(self):
    super(_XFlatGreenExtractorOp, self).__init__()

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
class _GreenRBExtractorOp(nn.Module):
  def __init__(self):
    super(_GreenRBExtractorOp, self).__init__()

  def forward(self, flat_green):
    # input: flat green channel 
    # output: green at Red and Blue
    flat_green_shape = list(flat_green.shape)
    N = flat_green_shape[0]
    quad_h = flat_green_shape[2] // 2
    quad_w = flat_green_shape[3] // 2
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
class _XGreenRBExtractorOp(nn.Module):
  def __init__(self):
    super(_XGreenRBExtractorOp, self).__init__()

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

 
"""
takes flat channel prediction and returns full 
channel in bayer quad format
"""
class _Flat2QuadOp(nn.Module):
  def __init__(self):
    super(_Flat2QuadOp, self).__init__()

  def forward(self, flat):
    # input: flat green channel 
    # output: green at Red and Blue
    flat_shape = list(flat.shape)
    N = flat_shape[0]
    quad_h = flat_shape[2] // 2
    quad_w = flat_shape[3] // 2
    out_shape = [N, 4, quad_h, quad_w]

    quad = torch.empty(torch.Size(out_shape), device=flat.device)
    quad[:,0,:,:] = flat[:,0,0::2,0::2]
    quad[:,1,:,:] = flat[:,0,0::2,1::2]
    quad[:,2,:,:] = flat[:,0,1::2,0::2]
    quad[:,3,:,:] = flat[:,0,1::2,1::2]

    return quad

 
class _SoftmaxOp(nn.Module):
  def __init__(self):
    super(_SoftmaxOp, self).__init__()
    self.f = nn.Softmax(dim=1)

  def forward(self, x):
    return self.f(x)


class _ReluOp(nn.Module):
  def __init__(self):
    super(_ReluOp, self).__init__()
    self.f = nn.ReLU()

  def forward(self, x):
    return self.f(x)


class _LogOp(nn.Module):
  def __init__(self):
    super(_LogOp, self).__init__()
  
  def forward(self, x):
    return torch.log(x)

class _ExpOp(nn.Module):
  def __init__(self):
    super(_ExpOp, self).__init__()
  
  def forward(self, x):
    return torch.exp(x)


class _LearnedDownsampleOp(nn.Module):
  def __init__(self, C_in, C_out, scale_factor, groups):
    super(_LearnedDownsampleOp, self).__init__()
    self.scale_factor = scale_factor
    self.downsample_w = scale_factor * 2 + (scale_factor % 2) # use odd kernel width for odd sampling factors 
    self.downsampler = nn.Conv2d(C_in, C_out, self.downsample_w, stride=scale_factor, groups=groups, padding=math.ceil((self.downsample_w-self.scale_factor)/2), bias=False)

  def forward(self, x):
    out = self.downsampler(x)
    if out.shape[2] != x.shape[2] / self.scale_factor:
      print(f'down input size {x.shape}')
      print(f'output shape {out.shape}')
    return out


"""
performs a unique convolution per spatial location within a period
"""
class _PeriodicConvOp(nn.Module):
  def __init__(self, C_in, C_out, period, kwidth):
    super(_PeriodicConvOp, self).__init__()
    self.period = period
    self.kwidth = kwidth

    self.im2col = nn.Unfold(kwidth, padding=kwidth//2)

    input_c = (period**2 * C_in * kwidth**2)
    output_c = (period**2 * C_out)
    self.conv = nn.Conv2d(input_c, output_c, 1, groups=self.period**2, padding=0, bias=False)
    self.unpack = nn.PixelShuffle(upscale_factor=period)


  def forward(self, x):
    tiles = self.im2col(x) # batch_size, c * kwidth**2, h * w
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
    
    conv_output = self.conv(packed) # batch_size, (c_out * period**2), h//period, w//period
    out = self.unpack(conv_output) # batch_size, c_out, h, w
    return out


class _PeriodicConvV2Op(nn.Module):
  def __init__(self, C_in, C_out, period, kwidth):
    super(_PeriodicConvV2Op, self).__init__()
    self.period = period
    self.kwidth = kwidth
    self.out_c = C_out 
    self.conv = nn.Conv2d(C_in, C_out*period**2, kwidth, padding=kwidth//2, bias=False)

  def forward(self, x):
    conv_out = self.conv(x)

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


class _PackOp(nn.Module):
  def __init__(self, factor):
    super(_PackOp, self).__init__()
    self.scale_factor = factor

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


class _LearnedPackOp(nn.Module):
  def __init__(self, C_in, C_out, factor):
    super(_LearnedPackOp, self).__init__()
    self.scale_factor = factor
    self.conv = nn.Conv2d(C_in, C_out, factor, stride=factor, bias=False, padding=0)

  def forward(self, x): 
    out = self.conv(x)
    return out


"""
Unpacks the mosaic
"""
class _UnpackOp(nn.Module):
  def __init__(self, C_in, factor):
    super(_UnpackOp, self).__init__()
    self.in_c = C_in
    self.scale_factor = factor
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor=factor)

  def forward(self, x):
    return self.pixel_shuffle(x)
    

class _BilinearUpsampleOp(nn.Module):
  def __init__(self, C_in, scale_factor):
    super(_BilinearUpsampleOp, self).__init__()
    self.in_c = C_in
    self.scale_factor = scale_factor
    self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

  def forward(self, x):
    out = self.upsampler(x)
    if out.shape[2] != x.shape[2] * self.scale_factor:
      print(f'up input shape {x.shape}')
      print(f'out shape {out.shape}')
    return out
    

class _LearnedUpsampleOp(nn.Module):
  def __init__(self, C_in, C_out, scale_factor, groups):
    super(_LearnedUpsampleOp, self).__init__()
    self.in_c = C_in
    self.out_c = C_out
    self.scale_factor = scale_factor
    self.upsampler = nn.ConvTranspose2d(self.in_c, self.out_c, scale_factor, groups=groups, stride=scale_factor)

  def forward(self, x):
    return self.upsampler(x)
 

class _Conv1x1Op(nn.Module):
  def __init__(self, C_in, C_out, groups):
    super(_Conv1x1Op, self).__init__()
    self.conv = nn.Conv2d(C_in, C_out, (1, 1), groups=groups, bias=False, padding=0)

  def forward(self, x): 
    out = self.conv(x)
    return out


# 1D diagonal convolution from top left corner to bottom right corner
class _DiagLRConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, padding):
    super(_DiagLRConv, self).__init__()
    self.padding = padding
    self.filter_w = nn.Parameter(torch.zeros(C_out, C_in, kernel_size))
   
  def forward(self, x):
    return nn.functional.conv2d(x, torch.diag_embed(self.filter_w), padding=self.padding)


# 1D diagonal convolution from top right corner to bottom left corner
class _DiagRLConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, padding):
    super(_DiagRLConv, self).__init__()
    self.padding = padding
    # self.mask = torch.zeros(C_out, C_in, kernel_size, kernel_size).cuda()
    self.filter_w = nn.Parameter(torch.zeros(C_out, C_in, kernel_size, kernel_size))
    self.mask = torch.zeros(C_out, C_in, kernel_size, kernel_size)
   
    for i in range(kernel_size):
      self.mask[..., i, kernel_size-i-1] = 1.0

  def forward(self, x):
    # if filter_w.is_cuda and not self.mask.is_cuda:
    #   self.mask = self.mask.to(device=f"cuda:{self.gpu_id}")
    return nn.functional.conv2d(x, (self.filter_w * self.mask), padding=self.padding)


class _Conv1DOp(nn.Module):
  def __init__(self, C_in, C_out, groups, kwidth):
    super(_Conv1DOp, self).__init__()
    num_vfilters = C_out // 2
    num_hfilters = C_out - num_vfilters
    self.v = nn.Conv2d(C_in, num_vfilters, (kwidth, 1), groups=groups, bias=False, padding=(kwidth//2, 0))
    self.h = nn.Conv2d(C_in, num_hfilters, (1, kwidth), groups=groups, bias=False, padding=(0, kwidth//2))
 
  def forward(self, x):
    return torch.cat((self.v(x), self.h(x)), 1)


class _Conv2DOp(nn.Module):
  def __init__(self, C_in, C_out, groups, kwidth):
    super(_Conv2DOp, self).__init__()
    self.conv = nn.Conv2d(C_in, C_out, (kwidth,kwidth), groups=groups, bias=False, padding=kwidth//2)

  def forward(self, x):
    return self.conv(x)


class _GroupedSumOp(nn.Module):
  def __init__(self, C_out):
    super(_GroupedSumOp, self).__init__()
    self.C_out = C_out

  def forward(self, x):
    x_shape = list(x.shape)
    x_reshaped = torch.reshape(x, (x_shape[0], self.C_out, x_shape[1]//self.C_out, x_shape[2], x_shape[3]))
    out = x_reshaped.sum(2, keepdim=False) 
    return out


class _InterleavedSumOp(nn.Module):
  def __init__(self, C_out):
    super(_InterleavedSumOp, self).__init__()
    self.C_out = C_out

  def forward(self, x):
    x_shape = list(x.shape)
    x_reshaped = torch.reshape(x, (x_shape[0], x_shape[1]//self.C_out, self.C_out, x_shape[2], x_shape[3]))
    out = x_reshaped.sum(1, keepdim=False) 
    return out


def set_parameters(model, calling_id, node2param):
  if "input" in calling_id:
    return
  node = getattr(model, calling_id)
  if calling_id in node2param:
    params = node2param[calling_id]

  if isinstance(node, _Conv1DOp):
    node.v.weight, node.v.bias = params[0], params[1]
    node.h.weight, node.h.bias = params[2], params[3]
  elif isinstance(node, _Conv1x1Op) or isinstance(node, _Conv2DOp):
    node.conv.weight, node.conv.bias = params[0], params[1]
  elif isinstance(node, _LearnedDownsampleOp):
    node.downsampler.weight, node.downsampler.bias = params[0], params[1]
  elif isinstance(node, _PeriodicConvOp) or isinstance(node, _PeriodicConvV2Op):
    node.conv.weight, node.conv.bias = params[0], params[1]
  elif isinstance(node, _LearnedPackOp):
    node.conv.weight, node.conv.bias = params[0], params[1]
  elif isinstance(node, _LearnedUpsampleOp):
    node.upsampler.weight, node.upsampler.bias = params[0], params[1]
