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
  out = SGreenExtractor(highres_green)
  
  out.assign_parents()
  out.compute_input_output_channels()
  compute_resolution(out)
  return out


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

  out = SGreenExtractor(highres_green)
  # predicing all the missing values
  out.assign_parents()
  out.compute_input_output_channels()
  compute_resolution(out)
  return out



def ChromaDemosaicknet(depth, width, no_grad, green_model, green_model_id):
  bayer = Input(4, name="Mosaic", resolution=float(1/2))
  rb = Input(2, name="RedBlueBayer", resolution=1/2)
  
  flat_green = Input(1, name="GreenExtractor", resolution=2, no_grad=True, node=green_model, green_model_id=green_model_id)

  green_quad = Flat2Quad(flat_green)
  green_quad.compute_input_output_channels()
  green_quad_input = Input(4, name="GreenQuad", resolution=1, node=green_quad, no_grad=True)

  green_rb = GreenRBExtractor(flat_green)
  green_rb.compute_input_output_channels()
  green_rb_input = Input(2, name="Green@RB", resolution=1, node=green_rb, no_grad=True)

  rb_upsampled = LearnedUpsample(rb, 2, factor=2)

  rb_min_g = Sub(rb_upsampled, green_rb_input)
  rb_min_g_stack_green = Stack(rb_min_g, green_quad_input)
  rb_min_g_stack_green.compute_input_output_channels()
  chroma_input = Input(6, name="RBdiffG_GreenQuad", resolution=1/2, no_grad=True, node=rb_min_g_stack_green)

  for i in range(depth):
    if i == 0:
      conv = Conv2D(chroma_input, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv)

  # unpack the learned residual twice
  # with a conv in between to ease the growing pains
  fullres_residual = LearnedUpsample(relu, width//2, factor=2)
  conv_residual = Conv2D(fullres_residual, width//2, kwidth=3)
  
  # upsample and unpack the bayer quad 
  # with a conv in between to ease the growing pains
  up1_bayer = LearnedUpsample(bayer, 4, factor=2)
  conv_up1bayer = Conv2D(up1_bayer, width//2, kwidth=3)
  up2_bayer = LearnedUpsample(conv_up1bayer, width//2, factor=2)

  # now we're in the upsampled image resolution
  stacked = Stack(up2_bayer, conv_residual) 

  highres_chroma_diff = Conv2D(stacked, 2, kwidth=3)
  chroma_pred = Add(highres_chroma_diff, flat_green)
  out = SRGBExtractor(flat_green, chroma_pred)
  
  out.assign_parents()
  out.compute_input_output_channels()
  compute_resolution(out)
  return out


def ChromaMultires(depth, width, no_grad, green_model, green_model_id):
  bayer = Input(4, name="Mosaic", resolution=float(1/2))
  rb = Input(2, name="RedBlueBayer", resolution=1/2)
  
  flat_green = Input(1, name="GreenExtractor", resolution=2, no_grad=True, node=green_model, green_model_id=green_model_id)

  green_quad = Flat2Quad(flat_green)
  green_quad.compute_input_output_channels()
  green_quad_input = Input(4, name="GreenQuad", resolution=1, node=green_quad, no_grad=True)

  green_rb = GreenRBExtractor(flat_green)
  green_rb.compute_input_output_channels()
  green_rb_input = Input(2, name="Green@RB", resolution=1, node=green_rb, no_grad=True)
  
  rb_upsampled = LearnedUpsample(rb, 2, factor=2)

  rb_min_g = Sub(rb_upsampled, green_rb_input)
  rb_min_g_stack_green = Stack(rb_min_g, green_quad_input)
  rb_min_g_stack_green.compute_input_output_channels()
  chroma_input = Input(6, name="RBdiffG_GreenQuad", resolution=1/2, no_grad=True, node=rb_min_g_stack_green)

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
  summed = GroupedSum(weighted_interps, width//2)

  # unpack the learned residual twice
  # with a conv in between to ease the growing pains
  fullres_residual = LearnedUpsample(summed, width//2, factor=2)
  conv_residual = Conv2D(fullres_residual, width//2, kwidth=3)

  # upsample and unpack the bayer quad 
  # with a conv in between to ease the growing pains
  up1_bayer = LearnedUpsample(bayer, 4, factor=2)
  conv_up1bayer = Conv2D(up1_bayer, width//2, kwidth=3)
  up2_bayer = LearnedUpsample(conv_up1bayer, width//2, factor=2)

  # now we're in the upsampled image resolution
  stacked = Stack(up2_bayer, conv_residual) 
  highres_chroma_diff = Conv2D(stacked, 2, kwidth=3)
  chroma_pred = Add(highres_chroma_diff, flat_green)

  out = SRGBExtractor(flat_green, chroma_pred)
  # predicing all the missing values
  out.assign_parents()
  out.compute_input_output_channels()
  compute_resolution(out)
  return out

