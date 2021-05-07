from demosaic_ast import *
from type_check import compute_resolution


def basic1D_green_model():
  # detector model
  bayer = Input(1, "Bayer")
  d1filters_d = Conv1D(bayer, 16)
  relu1 = Relu(d1filters_d)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  softmax = Softmax(fc2)

  # filter model
  bayer = Input(1, "Bayer")
  d1filters_f = Conv1D(bayer, 16)

  mul = Mul(d1filters_f, softmax)
  missing_green = SumR(mul)
  bayer = Input(1, "Bayer")
  full_green = GreenExtractor(missing_green, bayer)
  full_green.is_root = True
  full_green.assign_parents()
  full_green.compute_input_output_channels()

  return full_green


def basic2D_green_model():
  # detector model
  c = 16
  bayer = Input(1, "Bayer")
  dconv1 = Conv2D(bayer, c, kwidth=5)
  drelu1 = Relu(dconv1)
  dconv2 = Conv1x1(drelu1, c)
  drelu2 = Relu(dconv2)
  dconv3 = Conv1x1(drelu2, c)
  softmax = Softmax(dconv3)

  # filter model
  bayer = Input(1, "Bayer")
  fconv1 = Conv2D(bayer, c, kwidth=5)
  frelu1 = Relu(fconv1)
  fconv2 = Conv1x1(frelu1, c)
  frelu2 = Relu(fconv2)
  fconv3 = Conv1x1(frelu2, c)

  mul = Mul(fconv3, softmax)
  missing_green = SumR(mul)
  bayer = Input(1, "Bayer")
  full_green = GreenExtractor(missing_green, bayer)
  full_green.is_root = True
  full_green.assign_parents()
  full_green.compute_input_output_channels()

  return full_green


def basic2Dquad_green_model():
  # detector model
  bayer = QuadInput(4, "Bayer")
  d1filters_d = Conv2D(bayer, 16, kwidth=3)
  relu1 = Relu(d1filters_d)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  softmax = Softmax(fc2)

  # filter model
  bayer = QuadInput(4, "Bayer")
  d1filters_f = Conv2D(bayer, 16, kwidth=3)

  mul = Mul(d1filters_f, softmax)
  missing_green = Conv1x1(mul, 4)

  bayer = QuadInput(4, "Bayer")
  full_green = GreenQuadExtractor(missing_green, bayer)
  full_green.is_root = True
  full_green.assign_parents()
  full_green.compute_input_output_channels()

  return full_green


def ahd2D_green_model():
  bayer = Input(1, "Bayer");
  raw_convex = Conv2D(bayer, 16)
  convex = Softmax(raw_convex)
  interps = Conv2D(bayer, 16)
  mul = Mul(interps, convex)

  missing_green = SumR(mul)
  green = GreenExtractor(missing_green, bayer)
  green.assign_parents()
  green.compute_input_output_channels()
  return green


def ahd1D_green_model():
  bayer = Input(1, "Bayer");
  raw_convex = Conv1D(bayer, 16)
  convex = Softmax(raw_convex)
  interps = Conv1D(bayer, 16)
  mul = Mul(interps, convex)

  missing_green = SumR(mul)
  green = GreenExtractor(missing_green, bayer)
  green.assign_parents()
  green.compute_input_output_channels()
  return green


def multires2D_green_model():
  # detector model
  # lowres model
  bayer = Input(1, "Bayer")
  downsample = Downsample(bayer)
  selection_f_lowres = Conv2D(downsample, 16)
  relu1 = Relu(selection_f_lowres)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  upsample = Upsample(fc2)

  # fullres model
  bayer = Input(1, "Bayer")
  selection_f_fullres = Conv2D(bayer, 16)
  stack = Stack(upsample, selection_f_fullres)
  relu1 = Relu(stack)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  softmax = Softmax(fc2)

  # filter model
  interp_filters = Conv2D(bayer, 16)

  mul = Mul(interp_filters, softmax)
  missing_green = SumR(mul)
  green = GreenExtractor(missing_green, bayer)
  green.assign_parents()
  green.compute_input_output_channels()

  downsample.partner_set = set( [(upsample, id(upsample))] )
  upsample.partner_set = set( [(downsample, id(downsample))] )
  return green


def multires_green_model():
  # detector model
  # lowres model
  bayer = Input(1, "Bayer")
  downsample = Downsample(bayer)
  selection_f_lowres = Conv1D(downsample, 16)
  relu1 = Relu(selection_f_lowres)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  upsample = Upsample(fc2)

  # fullres model
  bayer = Input(1, "Bayer")
  selection_f_fullres = Conv1D(bayer, 16)
  stack = Stack(upsample, selection_f_fullres)
  relu1 = Relu(stack)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  softmax = Softmax(fc2)

  # filter model
  interp_filters = Conv1D(bayer, 16)

  mul = Mul(interp_filters, softmax)
  missing_green = SumR(mul)
  green = GreenExtractor(missing_green, bayer)
  green.assign_parents()
  green.compute_input_output_channels()

  downsample.partner_set = set( [ (upsample, id(upsample)) ] )
  upsample.partner_set = set( [ (downsample, id(downsample)) ] )

  return green


def fast_multires_green_model():
  # detector model
  # lowres model
  bayer = Input(1, "Bayer")
  downsample = Downsample(bayer)
  selection_f_lowres = Conv1D(downsample, 16)
  relu1 = Relu(selection_f_lowres)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  upsample = FastUpsample(fc2)

  # fullres model
  bayer = Input(1, "Bayer")
  selection_f_fullres = Conv1D(bayer, 16)
  stack = Stack(upsample, selection_f_fullres)
  relu1 = Relu(stack)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  softmax = Softmax(fc2)

  # filter model
  interp_filters = Conv1D(bayer, 16)

  mul = Mul(interp_filters, softmax)
  missing_green = SumR(mul)
  green = GreenExtractor(missing_green, bayer)
  green.assign_parents()
  green.compute_input_output_channels()

  downsample.partner_set = set( [ (upsample, id(upsample)) ] )
  upsample.partner_set = set( [ (downsample, id(downsample)) ] )

  return green


def multires_green_model2():
  # detector model
  # lowres model
  bayer = Input(1, "Bayer")
  downsample = Downsample(bayer)
  selection_f_lowres = Conv1D(downsample, 16)
  relu1 = Relu(selection_f_lowres)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  softmax = Softmax(fc2)
  upsample = Upsample(softmax)

  # fullres model
  bayer = Input(1, "Bayer")
  selection_f_fullres = Conv1D(bayer, 16)
  stack = Stack(upsample, selection_f_fullres)
  relu1 = Relu(stack)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  softmax = Softmax(fc2)

  # filter model
  interp_filters = Conv1D(bayer, 16)

  mul = Mul(interp_filters, softmax)
  missing_green = SumR(mul)
  green = GreenExtractor(missing_green, bayer)
  green.assign_parents()
  green.compute_input_output_channels()
  return green


def multires_green_model3():
  # detector model
  # lowres model
  bayer = Input(1, "Bayer")
  downsample = Downsample(bayer)
  selection_f_lowres = Conv1D(downsample, 16)
  relu1 = Relu(selection_f_lowres)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  upsample = Upsample(fc2)

  # fullres model
  bayer = Input(1, "Bayer")
  selection_f_fullres = Conv1D(bayer, 16)
  relu1 = Relu(selection_f_fullres)
  fc1 = Conv1x1(relu1, 16)
  stack = Stack(upsample, fc1)
  relu2 = Relu(stack)
  fc2 = Conv1x1(relu2, 16)
  softmax = Softmax(fc2)

  # filter model
  interp_filters = Conv1D(bayer, 16)

  mul = Mul(interp_filters, softmax)
  missing_green = SumR(mul)
  green = GreenExtractor(missing_green, bayer)
  green.assign_parents()
  green.compute_input_output_channels()

  return green


def full_model_greenInput():
  bayer = Input(1, "Bayer")
  green_input = Input(1, "Green")

  diff = Sub(bayer, green_input)

  h_conv1 = Conv2D(diff, 16)
  h_relu1 = Relu(h_conv1)
  chroma_h = Conv2D(h_relu1, 1)

  v_conv1 = Conv2D(diff, 16)
  v_relu1 = Relu(v_conv1)
  chroma_v = Conv2D(v_relu1, 1)

  q_conv1 = Conv2D(diff, 16)
  q_relu1 = Relu(q_conv1)
  chroma_q = Conv2D(q_relu1, 1)

  stack1 = Stack(chroma_h, chroma_v)
  chroma_diff = Stack(stack1, chroma_q)

  chroma_hvq = Add(chroma_diff, green_input)

  rgb = ChromaExtractor(chroma_hvq, bayer, green_input)
  rgb.assign_parents()
  rgb.compute_input_output_channels()

  return rgb 


def mini_demosaicnet():
  # fullres model
  bayer = Input(1, "Bayer")
  conv1 = Conv2D(bayer, 32, kwidth=3)
  relu1 = Relu(conv1)
  conv2 = Conv2D(relu1, 32, kwidth=3)
  relu2 = Relu(conv2)
  conv3 = Conv2D(relu2, 32, kwidth=3)
  relu3 = Relu(conv3)
  missing_green = Conv2D(relu3, 1, kwidth=3)
  
  green = GreenExtractor(missing_green, bayer)
  green.assign_parents()
  green.compute_input_output_channels()
  return green


"""
Uses one sequence of 3x3 convs on downsampled bayer quad and another on the fullres quad 
to produce lowres and full interpolations for green.
A third sequence of 3x3 convs on downsampled bayer quad is used to produce mixing weights for
the lowres and fullres interpolations.

chroma model: 
Weight predictor is on lowres r and b stacked with green predictions, learns a change of basis
in the first conv layer, then does subsequent series of convs 
Interp is on full res r - g / b - g stacked with the green predictions 
"""
def MultiresQuadGreenModel(depth, width):
  bayer = Input(4, name="Mosaic", resolution=float(1/2)) # quad Bayer
  downsampled_bayer = LearnedDownsample(bayer, width, factor=2)

  # lowres interp
  for i in range(depth):
    if i == 0:
      lowres_conv = Conv2D(downsampled_bayer, width, kwidth=3)
    else:
      lowres_conv = Conv2D(lowres_relu, width, kwidth=3)
    if i != depth - 1:
      lowres_relu = Relu(lowres_conv)

  lowres_interp = BilinearUpsample(lowres_conv, factor=2)
  lowres_interp.compute_input_output_channels()

  # fullres interp
  for i in range(depth):
    if i == 0:
      fullres_interp = Conv2D(bayer, width, kwidth=3)
    else:
      fullres_interp = Conv2D(fullres_relu, width, kwidth=3)
    if i != depth - 1:
      fullres_relu = Relu(fullres_interp)

  downsampled_bayer = LearnedDownsample(bayer, width, factor=2)

  # lowres weights
  for i in range(depth):
    if i == 0:
      weight_conv = Conv2D(downsampled_bayer, width, kwidth=3)
    else:
      weight_conv = Conv2D(weight_relu, width, kwidth=3)
    if i != depth - 1:
      weight_relu = Relu(weight_conv)
 
  upsampled_weights = BilinearUpsample(weight_conv, factor=2)
  weights = Softmax(upsampled_weights)
 
  lowres_mul = Mul(weights, lowres_interp)
  fullres_mul = Mul(weights, fullres_interp)

  lowres_sum = GroupedSum(lowres_mul, 2)
  fullres_sum = GroupedSum(fullres_mul, 2)

  bayer = Input(4, name="Mosaic", resolution=float(1/2))
  green_rb = Add(lowres_sum, fullres_sum)
  green = GreenExtractor(bayer, green_rb)
  green.assign_parents()
  green.compute_input_output_channels()
  compute_resolution(green)
  return green 


def GreenDemosaicknet(depth, width):
  bayer = Input(4, name="Mosaic", resolution=float(1/2))
  for i in range(depth):
    if i == 0:
      conv = Conv2D(bayer, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv)

  residual_prediction = Conv1x1(relu, 12)

  stacked = Stack(bayer, residual_prediction) 

  post_conv = Conv2D(stacked, width, kwidth=3)
  post_relu = Relu(post_conv)
  green_rb = Conv1x1(post_relu, 2)

  bayer = Input(4, name="Mosaic", resolution=float(1/2))
  green = GreenExtractor(bayer, green_rb)
  green.assign_parents()
  green.compute_input_output_channels()
  compute_resolution(green)
  return green


def XGreenDemosaicknet(depth, width):
  mosaic = Input(36, "Mosaic")
  for i in range(depth):
    if i == 0:
      conv = Conv1x1(mosaic, width)
    else:
      conv = Conv1x1(relu, width)
    relu = Relu(conv)

  residual_prediction = Conv1x1(relu, 16)

  stacked = Stack(mosaic, residual_prediction) 

  post_conv = Conv1x1(stacked, width)
  post_relu = Relu(post_conv)
  missing_green = Conv1x1(post_relu, 16)

  mosaic = Input(36, "Mosaic")
  green = XGreenExtractor(mosaic, missing_green)
  green.assign_parents()
  green.compute_input_output_channels()

  return green


def XFlatGreenDemosaicknet(depth, width):
  mosaic = Input(3, "Mosaic")
  for i in range(depth):
    if i == 0:
      conv = Conv2D(mosaic, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    relu = Relu(conv)

  stacked = Stack(mosaic, relu) 

  post_conv = Conv2D(stacked, width, kwidth=3)
  post_relu = Relu(post_conv)
  missing_green = Conv2D(post_relu, 1, kwidth=3)

  mosaic = Input(3, "Mosaic")
  green = XFlatGreenExtractor(mosaic, missing_green)
  green.assign_parents()
  green.compute_input_output_channels()

  return green



def GradientHalideModel(width, k):
  bayer = Input(4, "Bayer")
  
  weight_conv = Conv2D(bayer, width, kwidth=k)
  filter_conv = Conv2D(bayer, width, kwidth=k)
  weights = Softmax(weight_conv)

  mul = Mul(weights, filter_conv)
  green_rb = GroupedSum(mul, 2)

  bayer = Input(4, "Bayer")
  green = GreenExtractor(bayer, green_rb)
  
  green.assign_parents()
  green.compute_input_output_channels()

  return green


def Bilinear(width, k):
  bayer = Input(4, "Bayer")

  conv = Conv2D(bayer, width, kwidth=k)
  green_rb = GroupedSum(conv, 2)
  green = GreenExtractor(bayer, green_rb)
  green.assign_parents()
  green.compute_input_output_channels()

  return green


"""
Not to be confused with RGBGradientHalideModel which is the model
used in the gradient halide paper, this is the seed model we use
for predicting RGB all in one model
"""
def RGB8ChanGradientHalideModel(width, k):
  bayer = Input(4, "Bayer")
  
  weight_conv = Conv2D(bayer, width, kwidth=k)
  filter_conv = Conv2D(bayer, width, kwidth=k)
  weights = Softmax(weight_conv)

  mul = Mul(weights, filter_conv)
  rgb_pred = GroupedSum(mul, 8)

  bayer = Input(4, "Bayer")

  rgb = RGB8ChanExtractor(bayer, rgb_pred)
  rgb.assign_parents()
  rgb.compute_input_output_channels()

  return rgb


def RGBGradientHalideModel(width, k):
  bayer = Input(4, "Bayer")
  
  weight_conv = Conv2D(bayer, width, kwidth=k)
  filter_conv = Conv2D(bayer, width, kwidth=k)
  weights = Softmax(weight_conv)

  mul = Mul(weights, filter_conv)
  green_rb = GroupedSum(mul, 2)

  bayer = Input(4, "Bayer")
  flat_green = GreenExtractor(bayer, green_rb)

  green_rb = GreenRBExtractor(flat_green)
  green_quad = Flat2Quad(flat_green)

  rb = Input(2, "RedBlueBayer")
  rb_min_g = Sub(rb, green_rb)
  chroma_diff_pred = Conv2D(rb_min_g, 6, kwidth=3)
  
  green_GrGb = Input(2, "Green@GrGb")
  added_green = Stack(green_quad, green_GrGb) # Gr R B Gb Gr Gb --> predicted values R@Gr, B@R, R@B, R@Gb, B@Gr, B@Gb
  
  chroma_pred = Add(chroma_diff_pred, added_green) # Gr R B Gb Gr Gb  
  
  rgb = RGBExtractor(green_quad, rb, chroma_pred)
  rgb.assign_parents()
  rgb.compute_input_output_channels()

  return rgb


"""
lowres weights full res interp
"""
def ChromaSeedModel1(depth, width, no_grad, green_model, green_model_id):
  bayer = Input(4, "Bayer")
  green_GrGb = Input(2, "Green@GrGb")
  rb = Input(2, "RedBlueBayer")
  
  flat_green = Input(1, "GreenExtractor", no_grad=True, node=green_model, green_model_id=green_model_id)

  green_quad = Flat2Quad(flat_green)
  green_quad.compute_input_output_channels()
  green_quad_input = Input(4, "GreenQuad", node=green_quad, no_grad=True)

  green_rb = GreenRBExtractor(flat_green)
  green_rb.compute_input_output_channels()
  green_rb_input = Input(2, "Green@RB", node=green_rb, no_grad=True)

  rb_min_g = Sub(rb, green_rb_input)
  rb_min_g_stack_green = Stack(rb_min_g, green_quad_input)
  rb_min_g_stack_green.compute_input_output_channels()
  chroma_input = Input(6, "RBdiffG_GreenQuad", no_grad=True, node=rb_min_g_stack_green)
  
  # weight trunk
  downsampled = Downsample(chroma_input)
  for i in range(depth):
    if i == 0:
      conv = Conv2D(downsampled, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    if i != depth-1:
      relu = Relu(conv)

  upsampled_weights = Upsample(conv)
  weights = Softmax(upsampled_weights)
  weights.compute_input_output_channels()
  
  downsampled.partner_set = set( [(upsampled_weights, id(upsampled_weights))] )
  upsampled_weights.partner_set = set( [(downsampled, id(downsampled))] )

  # interp trunk
  for i in range(depth):
    if i == 0:
      interp = Conv2D(chroma_input, width, kwidth=3)
    else:
      interp = Conv2D(relu, width, kwidth=3)
    if i != depth-1:
      relu = Relu(interp)

  weighted_interps = Mul(weights, interp)
  chroma_diff_pred = GroupedSum(weighted_interps, 6) # 6 predicted values R@Gr, B@R, R@B, R@Gb, B@Gr, B@Gb

  added_green = Stack(green_quad_input, green_GrGb) # Gr R B Gb Gr Gb --> predicted values R@Gr, B@R, R@B, R@Gb, B@Gr, B@Gb
  chroma_pred = Add(chroma_diff_pred, added_green) # Gr R B Gb Gr Gb  

  rgb = RGBExtractor(green_quad_input, rb, chroma_pred)
  rgb.assign_parents()
  rgb.compute_input_output_channels()

  return rgb                                            


"""
lowres weights, lowres and full res interp
"""
def ChromaSeedModel2(depth, width, no_grad, green_model, green_model_id):

  bayer = Input(4, "Bayer")
  green_GrGb = Input(2, "Green@GrGb")
  rb = Input(2, "RedBlueBayer")
  
  flat_green = Input(1, "GreenExtractor", no_grad=True, node=green_model, green_model_id=green_model_id)

  green_quad = Flat2Quad(flat_green)
  green_quad.compute_input_output_channels()
  green_quad_input = Input(4, "GreenQuad", node=green_quad, no_grad=True)

  green_rb = GreenRBExtractor(flat_green)
  green_rb.compute_input_output_channels()
  green_rb_input = Input(2, "Green@RB", node=green_rb, no_grad=True)

  rb_min_g = Sub(rb, green_rb_input)
  rb_min_g_stack_green = Stack(rb_min_g, green_quad_input)
  rb_min_g_stack_green.compute_input_output_channels()
  chroma_input = Input(6, "RBdiffG_GreenQuad", no_grad=True, node=rb_min_g_stack_green)

  # weight trunk
  downsampled = Downsample(chroma_input)
  for i in range(depth):
    if i == 0:
      conv = Conv2D(downsampled, width, kwidth=3)
    else:
      conv = Conv2D(relu, width, kwidth=3)
    if i != depth-1:
      relu = Relu(conv)

  upsampled_weights = Upsample(conv)
  weights = Softmax(upsampled_weights)
  weights.compute_input_output_channels()
  
  downsampled.partner_set = set( [(upsampled_weights, id(upsampled_weights))] )
  upsampled_weights.partner_set = set( [(downsampled, id(downsampled))] )

  # lowres interp trunk
  downsampled = Downsample(chroma_input)
  for i in range(depth):
    if i == 0:
      lowres_conv = Conv2D(downsampled, width, kwidth=3)
    else:
      lowres_conv = Conv2D(relu, width, kwidth=3)
    if i != depth-1:
      relu = Relu(lowres_conv)

  lowres_interp = Upsample(lowres_conv)
  lowres_interp.compute_input_output_channels()

  downsampled.partner_set = set( [(lowres_interp, id(lowres_interp))] )
  lowres_interp.partner_set = set( [(downsampled, id(downsampled))] )

  # fullres interp trunk
  for i in range(depth):
    if i == 0:
      fullres_interp = Conv2D(chroma_input, width, kwidth=3)
    else:
      fullres_interp = Conv2D(relu, width, kwidth=3)
    if i != depth-1:
      relu = Relu(fullres_interp)

  lowres_mul = Mul(weights, lowres_interp)
  fullres_mul = Mul(weights, fullres_interp)

  lowres_sum = GroupedSum(lowres_mul, 6)
  fullres_sum = GroupedSum(fullres_mul, 6)
  
  chroma_diff_pred = Add(lowres_sum, fullres_sum)
  added_green = Stack(green_quad_input, green_GrGb) # Gr R B Gb Gr Gb --> predicted values R@Gr, B@R, R@B, R@Gb, B@Gr, B@Gb
  chroma_pred = Add(chroma_diff_pred, added_green) # Gr R B Gb Gr Gb  

  rgb = RGBExtractor(green_quad_input, rb, chroma_pred)
  rgb.assign_parents()
  rgb.compute_input_output_channels()

  return rgb         


def ChromaSeedModel3(depth, width, no_grad, green_model, green_model_id):
  bayer = Input(4, "Bayer")
  green_GrGb = Input(2, "Green@GrGb")
  rb = Input(2, "RedBlueBayer")
  
  flat_green = Input(1, "GreenExtractor", no_grad=True, node=green_model, green_model_id=green_model_id)

  green_quad = Flat2Quad(flat_green)
  green_quad.compute_input_output_channels()
  green_quad_input = Input(4, "GreenQuad", node=green_quad, no_grad=True)

  green_rb = GreenRBExtractor(flat_green)
  green_rb.compute_input_output_channels()
  green_rb_input = Input(2, "Green@RB", node=green_rb, no_grad=True)

  rb_min_g = Sub(rb, green_rb_input)
  rb_min_g_stack_green = Stack(rb_min_g, green_quad_input)
  rb_min_g_stack_green.compute_input_output_channels()
  chroma_input = Input(6, "RBdiffG_GreenQuad", no_grad=True, node=rb_min_g_stack_green)

  # interp trunk
  for i in range(depth):
    if i == 0:
      interp = Conv2D(chroma_input, width, kwidth=3)
    else:
      interp = Conv2D(relu, width, kwidth=3)
    if i != depth-1:
      relu = Relu(interp)

    residual_prediction = Conv1x1(relu, width)

  stacked = Stack(rb_min_g, residual_prediction) 

  post_conv = Conv2D(stacked, width, kwidth=3)
  post_relu = Relu(post_conv)
  chroma_diff_pred = Conv1x1(post_relu, 6) # 6 predicted values R@Gr, B@R, R@B, R@Gb, B@Gr, B@Gb
  
  added_green = Stack(green_quad_input, green_GrGb) # Gr R B Gb Gr Gb --> predicted values R@Gr, B@R, R@B, R@Gb, B@Gr, B@Gb
  chroma_pred = Add(chroma_diff_pred, added_green) # Gr R B Gb Gr Gb  

  rgb = RGBExtractor(green_quad_input, rb, chroma_pred)
  rgb.assign_parents()
  rgb.compute_input_output_channels()

  return rgb        


def ChromaGradientHalideModel(width, k, no_grad, green_model, green_model_id):
  bayer = Input(4, "Bayer")
  green_GrGb = Input(2, "Green@GrGb")
  rb = Input(2, "RedBlueBayer")
  
  flat_green = Input(1, "GreenExtractor", no_grad=True, node=green_model, green_model_id=green_model_id)

  green_quad = Flat2Quad(flat_green)
  green_quad.compute_input_output_channels()
  green_quad_input = Input(4, "GreenQuad", node=green_quad, no_grad=True)

  green_rb = GreenRBExtractor(flat_green)
  green_rb.compute_input_output_channels()
  green_rb_input = Input(2, "Green@RB", node=green_rb, no_grad=True)

  rb_min_g = Sub(rb, green_rb_input)
  rb_min_g_stack_green = Stack(rb_min_g, green_quad_input)
  rb_min_g_stack_green.compute_input_output_channels()
  chroma_input = Input(6, "RBdiffG_GreenQuad", no_grad=True, node=rb_min_g_stack_green)

  weight_conv = Conv2D(chroma_input, width, kwidth=k)
  filter_conv = Conv2D(chroma_input, width, kwidth=k)
  weights = Softmax(weight_conv)

  mul = Mul(weights, filter_conv)
  chroma_diff_pred = GroupedSum(mul, 6)

  added_green = Stack(green_quad_input, green_GrGb) # Gr R B Gb Gr Gb --> predicted values R@Gr, B@R, R@B, R@Gb, B@Gr, B@Gb
  chroma_pred = Add(chroma_diff_pred, added_green) # Gr R B Gb Gr Gb  

  rgb = RGBExtractor(green_quad_input, rb, chroma_pred)
  rgb.assign_parents()
  rgb.compute_input_output_channels()

  return rgb


def RGB8ChanDemosaicknet(depth, width):
  bayer = Input(4, name="Mosaic", resolution=float(1/2))
  # interp trunk
  for i in range(depth):
    if i == 0:
      interp = Conv2D(bayer, width, kwidth=3)
    else:
      interp = Conv2D(relu, width, kwidth=3)
    if i != depth-1:
      relu = Relu(interp)

    residual_prediction = Conv1x1(relu, width)
  
  bayer = Input(4, name="Mosaic", resolution=float(1/2))
  stacked = Stack(bayer, residual_prediction) 

  post_conv = Conv2D(stacked, width, kwidth=3)
  post_relu = Relu(post_conv)
  rgb_pred = Conv1x1(post_relu, 8) # 8 predicted missing values 

  bayer = Input(4, name="Mosaic", resolution=float(1/2))
  rgb = RGB8ChanExtractor(bayer, rgb_pred)
  rgb.assign_parents()
  rgb.compute_input_output_channels()

  return rgb        



def NASNet(depth, width):
  bayer = Input(3, name="Mosaic", resolution=float(1.0))
  # interp trunk
  for i in range(depth):
    if i == 0:
      interp = Conv2D(bayer, width, kwidth=3)
    else:
      interp = Conv2D(relu, width, kwidth=3)
    if i != depth-1:
      relu = Relu(interp)

    residual_prediction = Conv1x1(relu, width)
  
  bayer = Input(3, name="Mosaic", resolution=float(1.0))
  stacked = Stack(bayer, residual_prediction) 

  post_conv = Conv2D(stacked, width, kwidth=3)
  post_relu = Relu(post_conv)
  rgb_pred = Conv1x1(post_relu, 8) # 8 predicted missing values 

  bayer = Input(3, name="Mosaic", resolution=float(1.0))
  rgb = FlatRGB8ChanExtractor(bayer, rgb_pred)
  rgb.assign_parents()
  rgb.compute_input_output_channels()
  compute_resolution(rgb)

  return rgb        
