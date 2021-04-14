from demosaic_ast import *


def GreenDemosaicknet(depth, width):
  bayer = Input(4, "Mosaic")
  for i in range(depth):
    if i == 0:
      conv = Conv2D(bayer, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv)

  # unpack bayer and upsample
  flat_residual = Unpack(relu, factor=4)
  flat_bayer = Unpack(bayer, factor=2)
  upsampled_bayer = Upsample(flat_bayer)
  stacked = Stack(upsampled_bayer, flat_residual) 

  post_conv = Conv2D(stacked, width, kwidth=3)
  post_relu = Relu(post_conv)
  highres_green = Conv1x1(post_relu, 1)

  # predicing all the missing values
  highres_green.assign_parents()
  highres_green.compute_input_output_channels()

  return highres_green

