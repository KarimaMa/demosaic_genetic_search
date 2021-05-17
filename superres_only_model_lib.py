from demosaic_ast import *
from type_check import compute_resolution


def GreenNet(depth, width):
  image = Input(3, name="Image", resolution=1)
  for i in range(depth):
    if i == 0:
      conv = Conv2D(image, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv)

  # upsample the residual
  up1_residual = LearnedUpsample(relu, width//2, factor=2)
  conv_up1res = Conv2D(up1_residual, width//2, kwidth=3)
  
  # upsample and unpack the image
  up1_image = LearnedUpsample(image, 3, factor=2)
  conv_up1image = Conv2D(up1_image, width//2, kwidth=3)

  # now we're in the upsampled image resolution
  stacked = Stack(conv_up1image, conv_up1res) 

  highres_green = Conv2D(stacked, 1, kwidth=3)
  out = SGreenExtractor(highres_green)
  
  out.assign_parents()
  out.compute_input_output_channels()
  compute_resolution(out)
  return out


def GreenMultires(depth, width):
  image = Input(3, name="Image", resolution=1)
  for i in range(depth):
    if i == 0:
      conv = Conv2D(image, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv) 

  # weight trunk
  image = Input(3, name="Image", resolution=1)
  for i in range(depth):
    if i == 0:
      wconv = Conv2D(image, width, kwidth=3)
    else:
      wconv = Conv2D(wrelu, width, kwidth=3)
    wrelu = Relu(wconv)

  weights = Softmax(wrelu)
  weighted_interps = Mul(weights, relu)
  summed = GroupedSum(weighted_interps, 1)

  up1_residual = LearnedUpsample(summed, width//4, factor=2)
  conv_up1res = Conv2D(up1_residual, width//4, kwidth=3)

  # upsample and unpack the image quad 
  # with a conv in between to ease the growing pains
  up1_image = LearnedUpsample(image, 3, factor=2)
  conv_up1image = Conv2D(up1_image, width//4, kwidth=3)

  # now we're in the upsampled image resolution
  stacked = Stack(conv_up1image, conv_up1res) 
  highres_green = Conv2D(stacked, 1, kwidth=3)

  out = SGreenExtractor(highres_green)
  # predicing all the missing values
  out.assign_parents()
  out.compute_input_output_channels()
  compute_resolution(out)
  return out



def ChromaNet(depth, width, no_grad, green_model, green_model_id):
  image = Input(4, name="Image", resolution=1)
  
  superres_green = Input(1, name="GreenExtractor", resolution=2, no_grad=True, node=green_model, green_model_id=green_model_id)
  upsampled_image = LearnedUpsample(image, 3, factor=2)

  chroma_input = Stack(upsampled_image, superres_green)

  for i in range(depth):
    if i == 0:
      conv = Conv2D(chroma_input, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv)

  # stack the simple upsampled image and superres green with the predicted residual
  stacked = Stack(chroma_input, relu) 
  superres_chroma = Conv2D(stacked, 2, kwidth=3)
  out = SRGBExtractor(flat_green, superres_chroma)
  
  out.assign_parents()
  out.compute_input_output_channels()
  compute_resolution(out)
  return out


def ChromaMultires(depth, width, no_grad, green_model, green_model_id):
  image = Input(4, name="Image", resolution=1)
  superres_green = Input(1, name="GreenExtractor", resolution=2, no_grad=True, node=green_model, green_model_id=green_model_id)
  upsampled_image = LearnedUpsample(image, 3, factor=2)

  chroma_input = Stack(upsampled_image, superres_green)

  for i in range(depth):
    if i == 0:
      conv = Conv2D(chroma_input, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv) 

  # weight trunk
  for i in range(depth):
    if i == 0:
      wconv = Conv2D(chroma_input, width, kwidth=3)
    else:
      wconv = Conv2D(wrelu, width, kwidth=3)
    wrelu = Relu(wconv)

  weights = Softmax(wrelu)
  weighted_interps = Mul(weights, relu)
  summed = GroupedSum(weighted_interps, 2)

  # stack the simple upsampled image and superres green with the predicted residual
  stacked = Stack(chroma_input, summed) 
  # do a final conv
  superres_chroma = Conv2D(stacked, 2, kwidth=3)

  out = SRGBExtractor(flat_green, superres_chroma)
  # predicing all the missing values
  out.assign_parents()
  out.compute_input_output_channels()
  compute_resolution(out)
  return out


def RGBNet(depth, width):
  image = Input(3, name="Image", resolution=1)  
  upsampled_image = LearnedUpsample(image, 3, factor=2)

  for i in range(depth):
    if i == 0:
      conv = Conv2D(image, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv)

  residual = LearnedUpsample(relu, 3, factor=2)
  # stack the simple upsampled image and superres green with the predicted residual
  stacked = Stack(upsampled_image, residual) 
  post_conv = Conv2D(stacked, 3, kwidth=3)

  out = SExtractor(post_conv)
  
  out.assign_parents()
  out.compute_input_output_channels()
  compute_resolution(out)
  return out


def RGBWeighted(depth, width):
  image = Input(3, name="Image", resolution=1)
  upsampled_image = LearnedUpsample(image, 3, factor=2)

  for i in range(depth):
    if i == 0:
      conv = Conv2D(image, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv) 
  fresidual = LearnedUpsample(relu, 3, factor=2)


  # weight trunk
  for i in range(depth):
    if i == 0:
      wconv = Conv2D(image, width, kwidth=3)
    else:
      wconv = Conv2D(wrelu, width, kwidth=3)
    wrelu = Relu(wconv)

  wresidual = LearnedUpsample(wrelu, 3, factor=2)

  weights = Softmax(wresidual)
  weighted_interps = Mul(weights, fresidual)
  summed = GroupedSum(weighted_interps, 3)

  # stack the simple upsampled image and superres green with the predicted residual
  stacked = Stack(upsampled_image, summed) 
  # do a final conv
  post_conv = Conv2D(stacked, 3, kwidth=3)

  out = SExtractor(post_conv)
  # predicing all the missing values
  out.assign_parents()
  out.compute_input_output_channels()
  compute_resolution(out)
  return out


