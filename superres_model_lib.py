from demosaic_ast import *


def GreenDemosaicknet(depth, width):
  bayer = Input(4, "Mosaic")
  for i in range(depth):
    if i == 0:
      conv = Conv2D(bayer, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv)

  # unpack the learned residual twice
  # with a conv in between to ease the growing pains
  up1_residual = LearnedUpsample(relu, factor=2)
  conv_up1res = Conv2D(up1_residual, width//4, kwidth=3)
  up2_residual = LearnedUpsample(conv_up1res, factor=2)
  
  # upsample and unpack the bayer quad 
  # with a conv in between to ease the growing pains
  up1_bayer = Upsample(bayer)
  conv_up1bayer = Conv2D(up1_bayer, width//4, kwidth=3)
  up2_bayer = LearnedUpsample(conv_up1bayer, factor=2)

  # now we're in the upsampled image resolution
  stacked = Stack(up2_bayer, up2_residual) 

  highres_green = Conv2D(stacked, width//16, kwidth=3)
  # post_relu = Relu(post_conv)
  # highres_green = Conv1x1(post_relu, 1)

  # predicing all the missing values
  highres_green.assign_parents()
  highres_green.compute_input_output_channels()

  return highres_green

