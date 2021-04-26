from demosaic_ast import *


def XGreenDemosaicknet1(depth, width):
  mosaic = Input(36, "Mosaic3x3")
  # upsample to bayer quad level resolution
  higher_res = LearnedUpsample(mosaic, width, factor=2, groups=1)
  # pointwise = Conv1x1(higher_res, width)

  input_tensor = higher_res
  # main processign 
  for i in range(depth):
    conv = Conv2D(input_tensor, width, kwidth=3)
    relu = Relu(conv)
    input_tensor = relu

  stacked = Stack(higher_res, input_tensor) 
  residual_conv = Conv2D(stacked, width, kwidth=3)
  residual_relu = Relu(residual_conv)

  pointwise = Conv1x1(residual_relu, 4)
  # downsample back to xtrans resolution
  missing_green = Pack(pointwise, factor=2)
 
  mosaic = Input(36, "Mosaic")
  green = XGreenExtractor(mosaic, missing_green)
  green.assign_parents()
  green.compute_input_output_channels()

  return green


def XGreenDemosaicknet2(depth, width):
  mosaic = Input(36, "Mosaic3x3")
  # embed each 3x3 block of the 6x6 mosaic pattern stored as 
  # 9 consecutive values in the channel dimension
  embedded = Conv1x1(mosaic, width*4, groups=4)

  # unpack to sets of 2x2 blocks
  unpacked = Unpack(embedded, factor=2)

  # supposedly, unpacked should now be in a spatially invariant embedding
  input_tensor = unpacked
  # main processign 
  for i in range(depth):
    conv = Conv2D(input_tensor, width, kwidth=3)
    relu = Relu(conv)
    input_tensor = relu

  stacked = Stack(unpacked, input_tensor) 
  residual_conv = Conv2D(stacked, width, kwidth=3)
  residual_relu = Relu(residual_conv)

  pointwise = Conv1x1(residual_relu, 4)

  # downsample back to xtrans resolution
  missing_green = Pack(pointwise, factor=2)

  mosaic = Input(36, "Mosaic")
  green = XGreenExtractor(mosaic, missing_green)
  green.assign_parents()
  green.compute_input_output_channels()

  return green



# def XGreenDemosaicknet3(depth, width):
#   mosaic = Input(36, "Mosaic3x3")
#   # upsample to bayer quad level resolution
#   higher_res = LearnedUpsample(mosaic, width, factor=2, groups=1)
#   # pointwise = Conv1x1(higher_res, width)

#   input_tensor = higher_res
#   # main processign 
#   for i in range(depth):
#     conv = Conv2D(input_tensor, width, kwidth=3)
#     relu = Relu(conv)
#     input_tensor = relu

#   full_res = Unpack(input_tensor, factor=3)
#   flat_mosaic = Input(1, "FlatMosaic")

#   stacked = Stack(full_res, flat_mosaic) 
#   residual_conv = Conv2D(stacked, width//9, kwidth=3)
#   residual_relu = Relu(residual_conv)
#   missing_green = Conv1x1(residual_relu, 1)

#   mosaic3chan = Input(3, "Mosaic")
#   green = XFlatGreenExtractor(mosaic3chan, missing_green)
#   green.assign_parents()
#   green.compute_input_output_channels()

#   return green


def XGreenDemosaicknet3(depth, width):
  mosaic = Input(36, "Mosaic3x3")
  # upsample to bayer quad level resolution
  unpacked = Unpack(mosaic, factor=2)
  higher_res = PeriodicConv(unpacked, width, period=2, kwidth=1)
  # pointwise = Conv1x1(higher_res, width)

  input_tensor = higher_res
  # main processign 
  for i in range(depth):
    conv = Conv2D(input_tensor, width, kwidth=3)
    relu = Relu(conv)
    input_tensor = relu

  full_res = Unpack(input_tensor, factor=3)
  flat_mosaic = Input(1, "FlatMosaic")

  stacked = Stack(full_res, flat_mosaic) 
  residual_conv = Conv2D(stacked, width//9, kwidth=3)
  residual_relu = Relu(residual_conv)
  missing_green = Conv1x1(residual_relu, 1)

  mosaic3chan = Input(3, "Mosaic")
  green = XFlatGreenExtractor(mosaic3chan, missing_green)
  green.assign_parents()
  green.compute_input_output_channels()

  return green


"""
Same as dnet3 except it doesn't do any final processing on 
the full res level, instead it uses a residual connection with 
the input mosaic on the 3x3 grid level 
"""
def XGreenDemosaicknet4(depth, width):
  mosaic = Input(36, "Mosaic3x3")
  # upsample to bayer quad level resolution
  higher_res = LearnedUpsample(mosaic, width, factor=2, groups=1)
  # pointwise = Conv1x1(higher_res, width)

  input_tensor = higher_res
  # main processign 
  for i in range(depth):
    conv = Conv2D(input_tensor, width, kwidth=3)
    relu = Relu(conv)
    input_tensor = relu

  packed_mosaic = Unpack(mosaic, factor=2) # 3x3 grid packed mosaic, 9 channels
  stacked = Stack(input_tensor, packed_mosaic) # 9 + width channels
  residual_conv = Conv1x1(stacked, width) 
  residual_relu = Relu(residual_conv)

  flat = Unpack(residual_relu, factor=3) # back up to full res, 1 channel
  green_pred = Conv1x1(flat, 1) 

  mosaic3chan = Input(3, "Mosaic")
  green = XFlatGreenExtractor(mosaic3chan, green_pred)
  green.assign_parents()
  green.compute_input_output_channels()

  return green


"""
post processing on flat resolution but using periodic_conv method 
for translation invariance on the flat resolution
"""
def XGreenDemosaicknet5(depth, width):
  mosaic = Input(36, "Mosaic3x3")
  # upsample to bayer quad level resolution
  higher_res = LearnedUpsample(mosaic, width, factor=2, groups=1)
 
  input_tensor = higher_res

  # main processign 
  for i in range(depth):
    conv = Conv2D(input_tensor, width, kwidth=3)
    relu = Relu(conv)
    input_tensor = relu

  full_res = Unpack(input_tensor, factor=3)
  flat_mosaic = Input(1, "FlatMosaic")

  stacked = Stack(full_res, flat_mosaic) 

  periodic_conv = PeriodicConv(stacked, width//9, period=6, kwidth=3)
  post_relu = Relu(periodic_conv)
  missing_green = Conv1x1(post_relu, 1)

  mosaic3chan = Input(3, "Mosaic")
  green = XFlatGreenExtractor(mosaic3chan, missing_green)
  green.assign_parents()
  green.compute_input_output_channels()

  return green



"""
post processing on flat resolution using 36 non periodic convs
to see if the reason why period conv is worse is because of higher
parameter count
"""
def XGreenDemosaicknet6(depth, width):
  mosaic = Input(36, "Mosaic3x3")
  # upsample to bayer quad level resolution
  higher_res = LearnedUpsample(mosaic, width, factor=2, groups=1)
 
  input_tensor = higher_res

  # main processign 
  for i in range(depth):
    conv = Conv2D(input_tensor, width, kwidth=3)
    relu = Relu(conv)
    input_tensor = relu

  full_res = Unpack(input_tensor, factor=3)
  flat_mosaic = Input(1, "FlatMosaic")

  stacked = Stack(full_res, flat_mosaic) 

  post_conv = Conv2D(stacked, 36*(width//9), kwidth=3) # learning 36 convolutions
  post_relu = Relu(post_conv)
  missing_green = Conv1x1(post_relu, 1)

  mosaic3chan = Input(3, "Mosaic")
  green = XFlatGreenExtractor(mosaic3chan, missing_green)
  green.assign_parents()
  green.compute_input_output_channels()

  return green



"""
Color difference with post processing on flat resolution
"""

def XRGBDemosaicknet1(depth, width, no_grad, green_model, green_model_id):
  # set up our inputs
  rb_xtrans = Input(16, "RBXtrans")
  green_pred = Input(1, "GreenExtractor", no_grad=no_grad, node=green_model, green_model_id=green_model_id)
  packed_green = Pack(green_pred, factor=3) # pack it to the 3x3 grid 
  packed_green.compute_input_output_channels()
  packed_green_input = Input(9, "PackedGreen", node=packed_green, no_grad=no_grad)

  green_rb = XGreenRBExtractor(green_pred)
  green_rb.compute_input_output_channels()
  green_rb_input = Input(16, "Green@RB", node=green_rb, no_grad=no_grad)

  # subtract green at red and blue locations from red and blue values
  diff = Sub(rb_xtrans, green_rb_input)
  # upsample to 1/3 resolution where each pixel corresponds to a 3x3 block
  higher_res_diff = LearnedUpsample(diff, width, factor=2, groups=1) 
  # stack color diff with full green prediction ( both are now in the 3x3 block format )
  stacked = Stack(higher_res_diff, packed_green_input) # width + 9 channels 
  
  input_tensor = stacked
  # main processing
  for i in range(depth):
    conv = Conv2D(input_tensor, width, kwidth=3)
    relu = Relu(conv)
    input_tensor = relu

  # bring things up to full res
  flat_residual = Unpack(input_tensor, factor=3)
  
  flat_mosaic = Input(1, "FlatMosaic")
  xtrans_and_green = Stack(green_pred, flat_mosaic) 
  post_input = Stack(xtrans_and_green, flat_residual)

  post_conv = Conv2D(post_input, width//9, kwidth=3)
  post_relu = Relu(post_conv)
  chroma_diff_pred = Conv1x1(post_relu, 2) # red and blue values for each pixel location

  chroma_pred = Add(chroma_diff_pred, green_pred)
 
  mosaic3chan = Input(3, "Mosaic")
  chroma = XFlatRGBExtractor(green_pred, mosaic3chan, chroma_pred)
  chroma.assign_parents()
  chroma.compute_input_output_channels()

  return chroma


"""
Color differencing with post processing on 3x3 grid
"""

def XRGBDemosaicknet2(depth, width, no_grad, green_model, green_model_id):
  # set up our inputs
  rb_xtrans = Input(16, "RBXtrans")
  green_pred = Input(1, "GreenExtractor", no_grad=no_grad, node=green_model, green_model_id=green_model_id)
  packed_green = Pack(green_pred, factor=3) # pack it to the 3x3 grid 
  packed_green.compute_input_output_channels()
  packed_green_input = Input(9, "PackedGreen", node=packed_green, no_grad=no_grad)

  green_rb = XGreenRBExtractor(green_pred)
  green_rb.compute_input_output_channels()
  green_rb_input = Input(16, "Green@RB", node=green_rb, no_grad=no_grad)

  # subtract green at red and blue locations from red and blue values
  diff = Sub(rb_xtrans, green_rb_input)
  # upsample to 1/3 resolution where each pixel corresponds to a 3x3 block
  higher_res_diff = LearnedUpsample(diff, width, factor=2, groups=1) 
  # stack color diff with full green prediction ( both are now in the 3x3 block format )
  stacked = Stack(higher_res_diff, packed_green_input) # width + 9 channels 
  
  input_tensor = stacked
  # main processing
  for i in range(depth):
    if i == 0:
      conv = Conv1x1(input_tensor, width)
    else:
      conv = Conv2D(input_tensor, width, kwidth=3)
    relu = Relu(conv)
    input_tensor = relu

  up_rb_xtrans = Unpack(rb_xtrans, factor=2)
  up_rb_green = Stack(up_rb_xtrans, packed_green) # all xtrans info plus predicted green
  post_input = Stack(input_tensor, up_rb_green) 

  post_conv = Conv2D(post_input, width, kwidth=3)
  post_relu = Relu(post_conv)
  chroma_diff_pred = Conv1x1(post_relu, 18)
  flat_chroma_diff_pred = Unpack(chroma_diff_pred, factor=3)

  # bring things up to full res
  chroma_pred = Add(flat_chroma_diff_pred, green_pred)

  mosaic3chan = Input(3, "Mosaic")
  chroma = XFlatRGBExtractor(green_pred, mosaic3chan, chroma_pred)
  chroma.assign_parents()
  chroma.compute_input_output_channels()

  return chroma


"""
No color difference, post process on 3x3 grid resolution
"""
def XRGBDemosaicknet3(depth, width, no_grad, green_model, green_model_id):
  # set up our inputs
  rb_xtrans = Input(16, "RBXtrans")
  green_pred = Input(1, "GreenExtractor", no_grad=no_grad, node=green_model, green_model_id=green_model_id)
  packed_green = Pack(green_pred, factor=3) # pack it to the 3x3 grid 
  packed_green.compute_input_output_channels()
  packed_green_input = Input(9, "PackedGreen", node=packed_green, no_grad=no_grad)

  up_rb_xtrans = LearnedUpsample(rb_xtrans, width, factor=2, groups=1) 
  stacked = Stack(up_rb_xtrans, packed_green_input) # width + 9 channels 
  
  input_tensor = stacked
  # main processing
  for i in range(depth):
    if i == 0:
      conv = Conv1x1(input_tensor, width)
    else:
      conv = Conv2D(input_tensor, width, kwidth=3)
    relu = Relu(conv)
    input_tensor = relu

  up_rb_xtrans = Unpack(rb_xtrans, factor=2)
  up_rb_green = Stack(up_rb_xtrans, packed_green) # all xtrans info plus predicted green
  residual_stacked = Stack(input_tensor, up_rb_green) 

  post_conv = Conv2D(residual_stacked, width, kwidth=3)
  post_relu = Relu(post_conv)
  chroma_pred = Conv1x1(post_relu, 18)
  flat_chroma_pred = Unpack(chroma_pred, factor=3)

  mosaic3chan = Input(3, "Mosaic")
  chroma = XFlatRGBExtractor(green_pred, mosaic3chan, flat_chroma_pred)
  chroma.assign_parents()
  chroma.compute_input_output_channels()

  return chroma






def XFlatGreenDemosaicknet(depth, width):
  flat_mosaic = Input(1, "FlatMosaic")
  for i in range(depth):
    if i == 0:
      conv = Conv2D(flat_mosaic, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv)

  stacked = Stack(flat_mosaic, relu) 

  post_conv = Conv2D(stacked, width, kwidth=3)
  post_relu = Relu(post_conv)
  missing_green = Conv2D(post_relu, 1, kwidth=3)

  mosaic = Input(3, "Mosaic")
  green = XFlatGreenExtractor(mosaic, missing_green)
  green.assign_parents()
  green.compute_input_output_channels()

  return green



