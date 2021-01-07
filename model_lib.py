from demosaic_ast import *

def updown_test_model():
  bayer = Input(1, "Bayer")
  down = Downsample(bayer)
  up = Upsample(down)
  return up

def fastupdown_test_model():
  bayer = Input(1, "Bayer")
  down = Downsample(bayer)
  up = FastUpsample(down)
  return up

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


def basic2D_l5_green_model():
  # detector model
  c = 4
  bayer = Input(1, "Bayer")
  dconv1 = Conv2D(bayer, c, kwidth=3)
  drelu1 = Relu(dconv1)
  dconv2 = Conv2D(drelu1, c, kwidth=3)
  drelu2 = Relu(dconv2)
  dconv3 = Conv2D(drelu2, c, kwidth=3)
  drelu3 = Relu(dconv3)
  dconv4 = Conv2D(drelu3, c, kwidth=3)
  drelu4 = Relu(dconv4)
  dconv5 = Conv2D(drelu4, c, kwidth=3)
  softmax = Softmax(dconv5)

  # filter model
  bayer = Input(1, "Bayer")
  fconv1 = Conv2D(bayer, c, kwidth=3)
  frelu1 = Relu(fconv1)
  fconv2 = Conv2D(frelu1, c, kwidth=3)
  frelu2 = Relu(fconv2)
  fconv3 = Conv2D(frelu2, c, kwidth=3)
  frelu3 = Relu(fconv3)
  fconv4 = Conv2D(frelu3, c, kwidth=3)
  frelu4 = Relu(fconv4)
  fconv5 = Conv2D(frelu4, c, kwidth=3)

  mul = Mul(fconv5, softmax)
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


def red_gr_diff_model():
  # Red at Gr
  print("GR diff model")
  gr_bayer = Input(1, "Bayer")
  green = Input(1, "Green")
  diff = Sub(gr_bayer, green)
  gr_convd = Conv2D(diff, 16)
  gr_relu1 = Relu(gr_convd)
  gr_fc1 = Conv1x1(gr_relu1, 16)
  gr_relu2 = Relu(gr_fc1)
  gr_fc2 = Conv1x1(gr_relu2, 16)
  gr_softmax = Softmax(gr_fc2)

  # filter model
  gr_bayer = Input(1, "Bayer")
  gr_convf = Conv2D(gr_bayer, 16)

  gr_mul = Mul(gr_convf, gr_softmax)
  R_at_Gr_diff = SumR(gr_mul)

  R_at_Gr = Add(R_at_Gr_diff, green)

  red = RGrExtractor(R_at_Gr)
  red.assign_parents()
  red.compute_input_output_channels()

  return red

def red_gb_diff_model():
  # Red at Gr
  print("GB diff model")
  gb_bayer = Input(1, "Bayer")
  green = Input(1, "Green")
  diff = Sub(gb_bayer, green)
  gb_convd = Conv2D(diff, 16)
  gb_relu1 = Relu(gb_convd)
  gb_fc1 = Conv1x1(gb_relu1, 16)
  gb_relu2 = Relu(gb_fc1)
  gb_fc2 = Conv1x1(gb_relu2, 16)
  gb_softmax = Softmax(gb_fc2)

  # filter model
  gb_bayer = Input(1, "Bayer")
  gb_convf = Conv2D(gb_bayer, 16)

  gb_mul = Mul(gb_convf, gb_softmax)
  R_at_Gb_diff = SumR(gb_mul)

  R_at_Gb = Add(R_at_Gb_diff, green)

  red = RGbExtractor(R_at_Gb)
  red.assign_parents()
  red.compute_input_output_channels()

  return red

def red_b_diff_model():
  # Red at B
  print("B diff model")
  b_bayer = Input(1, "Bayer")
  green = Input(1, "Green")
  diff = Sub(b_bayer, green)
  b_convd = Conv2D(diff, 16)
  b_relu1 = Relu(b_convd)
  b_fc1 = Conv1x1(b_relu1, 16)
  b_relu2 = Relu(b_fc1)
  b_fc2 = Conv1x1(b_relu2, 16)
  b_softmax = Softmax(b_fc2)

  # filter model
  b_bayer = Input(1, "Bayer")
  b_convf = Conv2D(b_bayer, 16)

  b_mul = Mul(b_convf, b_softmax)
  R_at_B_diff = SumR(b_mul)

  R_at_B = Add(R_at_B_diff, green)

  red = RBExtractor(R_at_B)
  red.assign_parents()
  red.compute_input_output_channels()

  return red


def red_gr_simple_model():
  # Red at Gr
  gr_bayer = Input(1, "Bayer")
  gr_convd = Conv2D(gr_bayer, 16)
  gr_relu1 = Relu(gr_convd)
  gr_fc1 = Conv1x1(gr_relu1, 16)
  gr_relu2 = Relu(gr_fc1)
  gr_fc2 = Conv1x1(gr_relu2, 16)
  gr_softmax = Softmax(gr_fc2)

  # filter model
  gr_bayer = Input(1, "Bayer")
  gr_convf = Conv2D(gr_bayer, 16)

  gr_mul = Mul(gr_convf, gr_softmax)
  R_at_Gr = SumR(gr_mul)

  red = RGrExtractor(R_at_Gr)
  red.assign_parents()
  red.compute_input_output_channels()

  return red


def red_gb_simple_model():
  # Red at Gb
  gb_bayer = Input(1, "Bayer")
  gb_convd = Conv2D(gb_bayer, 16)
  gb_relu1 = Relu(gb_convd)
  gb_fc1 = Conv1x1(gb_relu1, 16)
  gb_relu2 = Relu(gb_fc1)
  gb_fc2 = Conv1x1(gb_relu2, 16)
  gb_softmax = Softmax(gb_fc2)

  # filter model
  gb_bayer = Input(1, "Bayer")
  gb_convf = Conv2D(gb_bayer, 16)

  gb_mul = Mul(gb_convf, gb_softmax)
  R_at_Gb = SumR(gb_mul)

  red = RGbExtractor(R_at_Gb)
  red.assign_parents()
  red.compute_input_output_channels()

  return red


def red_b_simple_model():
  # Red at B
  b_bayer = Input(1, "Bayer")
  b_convd = Conv2D(b_bayer, 16)
  b_relu1 = Relu(b_convd)
  b_fc1 = Conv1x1(b_relu1, 16)
  b_relu2 = Relu(b_fc1)
  b_fc2 = Conv1x1(b_relu2, 16)
  b_softmax = Softmax(b_fc2)

  # filter model
  b_bayer = Input(1, "Bayer")
  b_convf = Conv2D(b_bayer, 16)

  b_mul = Mul(b_convf, b_softmax)
  R_at_B = SumR(b_mul)

  red = RBExtractor(R_at_B)
  red.assign_parents()
  red.compute_input_output_channels()

  return red

def blue_gr_diff_model():
  # Red at Gr
  print("GR diff model")
  gr_bayer = Input(1, "Bayer")
  green = Input(1, "Green")
  diff = Sub(gr_bayer, green)
  gr_convd = Conv2D(diff, 16)
  gr_relu1 = Relu(gr_convd)
  gr_fc1 = Conv1x1(gr_relu1, 16)
  gr_relu2 = Relu(gr_fc1)
  gr_fc2 = Conv1x1(gr_relu2, 16)
  gr_softmax = Softmax(gr_fc2)

  # filter model
  gr_bayer = Input(1, "Bayer")
  gr_convf = Conv2D(gr_bayer, 16)

  gr_mul = Mul(gr_convf, gr_softmax)
  B_at_Gr_diff = SumR(gr_mul)

  B_at_Gr = Add(B_at_Gr_diff, green)

  blue = BGrExtractor(B_at_Gr)
  blue.assign_parents()
  blue.compute_input_output_channels()

  return blue

def blue_gb_diff_model():
  # Red at Gr
  print("GB diff model")
  gb_bayer = Input(1, "Bayer")
  green = Input(1, "Green")
  diff = Sub(gb_bayer, green)
  gb_convd = Conv2D(diff, 16)
  gb_relu1 = Relu(gb_convd)
  gb_fc1 = Conv1x1(gb_relu1, 16)
  gb_relu2 = Relu(gb_fc1)
  gb_fc2 = Conv1x1(gb_relu2, 16)
  gb_softmax = Softmax(gb_fc2)

  # filter model
  gb_bayer = Input(1, "Bayer")
  gb_convf = Conv2D(gb_bayer, 16)

  gb_mul = Mul(gb_convf, gb_softmax)
  B_at_Gb_diff = SumR(gb_mul)

  B_at_Gb = Add(B_at_Gb_diff, green)

  blue = BGbExtractor(B_at_Gb)
  blue.assign_parents()
  blue.compute_input_output_channels()

  return blue

def blue_r_diff_model():
  # Red at B
  print("R diff model")
  r_bayer = Input(1, "Bayer")
  green = Input(1, "Green")
  diff = Sub(r_bayer, green)
  r_convd = Conv2D(diff, 16)
  r_relu1 = Relu(r_convd)
  r_fc1 = Conv1x1(r_relu1, 16)
  r_relu2 = Relu(r_fc1)
  r_fc2 = Conv1x1(r_relu2, 16)
  r_softmax = Softmax(r_fc2)

  # filter model
  r_bayer = Input(1, "Bayer")
  r_convf = Conv2D(r_bayer, 16)

  r_mul = Mul(r_convf, r_softmax)
  B_at_R_diff = SumR(r_mul)

  B_at_R = Add(B_at_R_diff, green)

  blue = BRExtractor(B_at_R)
  blue.assign_parents()
  blue.compute_input_output_channels()

  return blue


def red_quad_diff_model():
  # Red at B
  print("Red Quad diff model")
  bayer = Input(4, "BayerQuad") # quad Bayer
  green = Input(4, "GreenQuad") # quad Green

  diff = Sub(bayer, green)
  conv1 = Conv2D(diff, 32, kwidth=3)
  relu1 = Relu(conv1)
  conv2 = Conv2D(relu1, 32, kwidth=3)
  relu2 = Relu(conv2)
  conv3 = Conv1x1(relu2, 32)
  relu3 = Relu(conv3)
  chroma_diff = Conv1x1(relu3, 4)

  chroma = Add(chroma_diff, green)

  out = ChromaExtractor(chroma, bayer)
  out.assign_parents()
  out.compute_input_output_channels()

  return out


def redblue_quad_diff_model():
  # Red at B
  print("Red Blue Quad diff model")
  bayer = QuadInput(4, "Bayer") # quad Bayer
  green = QuadInput(4, "Green") # quad Green

  diff = Sub(bayer, green)
  conv1 = Conv2D(diff, 32, kwidth=3)
  relu1 = Relu(conv1)
  conv2 = Conv2D(relu1, 32, kwidth=3)
  relu2 = Relu(conv2)
  conv3 = Conv1x1(relu2, 32)
  relu3 = Relu(conv3)
  chroma_diff = Conv1x1(relu3, 8)

  green = QuadInput(4, "Green") # quad Green
  chroma = Add(chroma_diff, green)

  out = RBQuadExtractor(chroma, bayer)
  out.assign_parents()
  out.compute_input_output_channels()

  return out


def rgb_quad_diff_model():
  # Red at B
  c = 16
  print("Red Green Blue Quad diff model")
  bayer = QuadInput(4, "Bayer") # quad Bayer
  green = QuadInput(4, "Green") # quad Green

  diff = Sub(bayer, green)
  conv1 = Conv2D(diff, c, kwidth=3)
  relu1 = Relu(conv1)
  conv2 = Conv2D(relu1, c, kwidth=3)
  relu2 = Relu(conv2)
  conv3 = Conv2D(relu2, c, kwidth=3)
  relu3 = Relu(conv3)
  chroma_diff = Conv1x1(relu3, 8)

  green = QuadInput(4, "Green") # quad Green
  chroma = Add(chroma_diff, green)

  green = QuadInput(4, "Green") # quad Green
  out = RGBQuadExtractor(chroma, bayer, green)
  out.assign_parents()
  out.compute_input_output_channels()

  return out


def rgb2_quad_diff_model():
  # Red at B
  print("Red Green Blue Quad diff model")
  bayer = QuadInput(4, "Bayer") # quad Bayer
  green = QuadInput(4, "Green") # quad Green

  diff = Sub(bayer, green)
  conv1 = Conv2D(diff, 47, kwidth=3)
  relu1 = Relu(conv1)
  conv2 = Conv2D(relu1, 47, kwidth=3)
  relu2 = Relu(conv2)
  conv3 = Conv1x1(relu2, 47)
  relu3 = Relu(conv3)
  chroma_diff = Conv1x1(relu3, 12)

  green = QuadInput(4, "Green") # quad Green
  chroma = Add(chroma_diff, green)

  bayer = QuadInput(4, "Bayer") # quad Bayer
  out = RGB2QuadExtractor(chroma, bayer)
  out.assign_parents()
  out.compute_input_output_channels()

  return out


def rgb_mul_quad_diff_model():
  # Red at B
  print("RGB MUL Quad diff model")
  bayer = QuadInput(4, "Bayer") # quad Bayer
  green = QuadInput(4, "Green") # quad Green

  diff = Sub(bayer, green)
  conv1 = Conv2D(diff, 24, kwidth=3)
  relu1 = Relu(conv1)
  conv2 = Conv2D(relu1, 24, kwidth=3)
  relu2 = Relu(conv2)
  interp = Conv1x1(relu2, 24)
 
  w_conv1 = Conv2D(bayer, 24, kwidth=3)
  w_relu1 = Relu(w_conv1)
  w_conv2 = Conv2D(w_relu1, 24, kwidth=3)
  w_relu2 = Relu(w_conv2)
  w_conv3 = Conv1x1(w_relu2, 24)
  weights = Softmax(w_conv3)

  mul = Mul(weights, interp)
  chroma_diff = Conv1x1(mul, 12)
  green = QuadInput(4, "Green") # quad Green
  chroma = Add(chroma_diff, green)

  green = QuadInput(4, "Green") # quad Green
  bayer = QuadInput(4, "Bayer") # quad Bayer
  out = RGBQuadExtractor(chroma, bayer, green)

  out.assign_parents()
  out.compute_input_output_channels()

  return out


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
  bayer = Input(4, "Bayer") # quad Bayer
  downsampled_bayer = Downsample(bayer)

  # lowres interp
  for i in range(depth):
    if i == 0:
      lowres_conv = Conv2D(downsampled_bayer, width, kwidth=3)
    else:
      lowres_conv = Conv2D(lowres_relu, width, kwidth=3)
    if i != depth - 1:
      lowres_relu = Relu(lowres_conv)

  lowres_interp = Upsample(lowres_conv)
  lowres_interp.compute_input_output_channels()

  downsampled_bayer.partner_set = set( [(lowres_interp, id(lowres_interp))] )
  lowres_interp.partner_set = set( [(downsampled_bayer, id(downsampled_bayer))] )

  # fullres interp
  for i in range(depth):
    if i == 0:
      fullres_interp = Conv2D(bayer, width, kwidth=3)
    else:
      fullres_interp = Conv2D(fullres_relu, width, kwidth=3)
    if i != depth - 1:
      fullres_relu = Relu(fullres_interp)

  downsampled_bayer = Downsample(bayer)
  # lowres weights
  for i in range(depth):
    if i == 0:
      weight_conv = Conv2D(downsampled_bayer, width, kwidth=3)
    else:
      weight_conv = Conv2D(weight_relu, width, kwidth=3)
    if i != depth - 1:
      weight_relu = Relu(weight_conv)
 
  upsampled_weights = Upsample(weight_conv)
  weights = Softmax(upsampled_weights)
  weights.compute_input_output_channels()
  
  downsampled_bayer.partner_set = set( [(upsampled_weights, id(upsampled_weights))] )
  upsampled_weights.partner_set = set( [(downsampled_bayer, id(downsampled_bayer))] )

  lowres_mul = Mul(weights, lowres_interp)
  fullres_mul = Mul(weights, fullres_interp)

  lowres_sum = GroupedSum(lowres_mul, 2)
  fullres_sum = GroupedSum(fullres_mul, 2)

  bayer = Input(4, "Bayer")
  green_rb = Add(lowres_sum, fullres_sum)
  green = GreenExtractor(bayer, green_rb)
  green.assign_parents()
  green.compute_input_output_channels()

  return green 

 
def GreenDemosaicknet(depth, width):
  bayer = Input(4, "Bayer")
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

  bayer = Input(4, "Bayer")
  green = GreenExtractor(bayer, green_rb)
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

  rb_min_g = Sub(rb, green_rb)
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

  rb_min_g = Sub(rb, green_rb)
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

  rb_min_g = Sub(rb, green_rb)
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

    residual_prediction = Conv1x1(relu, 6)

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
