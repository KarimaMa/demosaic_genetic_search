from demosaic_ast import *
from type_check import compute_resolution


def GreenDemosaicknet(depth, width):
  bayer = Input(4, name="Mosaic", resolution=float(1/2))
  for i in range(depth):
    if i == 0:
      conv = Conv2D(bayer, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv)

  # unpack the learned residual twice
  # with a conv in between to ease the growing pains
  up1_residual = LearnedUpsample(relu, width//2, factor=2)
  conv_up1res = Conv2D(up1_residual, width//2, kwidth=3)
  up2_residual = LearnedUpsample(conv_up1res, width//4, factor=2)
  
  # upsample and unpack the bayer quad 
  # with a conv in between to ease the growing pains
  up1_bayer = LearnedUpsample(bayer, 4, factor=2)
  conv_up1bayer = Conv2D(up1_bayer, width//2, kwidth=3)
  up2_bayer = LearnedUpsample(conv_up1bayer, width//4, factor=2)

  # now we're in the upsampled image resolution
  stacked = Stack(up2_bayer, up2_residual) 

  highres_green = Conv2D(stacked, 1, kwidth=3)
  # post_relu = Relu(post_conv)
  # highres_green = Conv1x1(post_relu, 1)

  # predicing all the missing values
  highres_green.assign_parents()
  highres_green.compute_input_output_channels()
  compute_resolution(highres_green)
  return highres_green


def GreenMultires(depth, width):
  bayer = Input(4, name="Mosaic", resolution=float(1/2))
  for i in range(depth):
    if i == 0:
      conv = Conv2D(bayer, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv) 

  # weight trunk
  bayer = Input(4, name="Mosaic", resolution=float(1/2))
  for i in range(depth):
    if i == 0:
      wconv = Conv2D(bayer, width, kwidth=3)
    else:
      wconv = Conv2D(wrelu, width, kwidth=3)
    wrelu = Relu(wconv)

  weights = Softmax(wrelu)
  weighted_interps = Mul(weights, relu)
  summed = GroupedSum(weighted_interps, width//4)

  # unpack the learned residual twice
  # with a conv in between to ease the growing pains
  up1_residual = LearnedUpsample(summed, width//4, factor=2)
  conv_up1res = Conv2D(up1_residual, width//4, kwidth=3)
  up2_residual = LearnedUpsample(conv_up1res, width//4, factor=2)

  # upsample and unpack the bayer quad 
  # with a conv in between to ease the growing pains
  up1_bayer = LearnedUpsample(bayer, 4, factor=2)
  conv_up1bayer = Conv2D(up1_bayer, width//4, kwidth=3)
  up2_bayer = LearnedUpsample(conv_up1bayer, width//4, factor=2)

  # now we're in the upsampled image resolution
  stacked = Stack(up2_bayer, up2_residual) 
  highres_green = Conv2D(stacked, 1, kwidth=3)

  # predicing all the missing values
  highres_green.assign_parents()
  highres_green.compute_input_output_channels()
  compute_resolution(highres_green)
  return highres_green


