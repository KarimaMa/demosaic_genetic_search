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
  bayer = Input(1, "Bayer")
  d1filters_d = Conv2D(bayer, 16)
  relu1 = Relu(d1filters_d)
  fc1 = Conv1x1(relu1, 16)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 16)
  softmax = Softmax(fc2)

  # filter model
  bayer = Input(1, "Bayer")
  d1filters_f = Conv2D(bayer, 16)

  mul = Mul(d1filters_f, softmax)
  missing_green = SumR(mul)
  bayer = Input(1, "Bayer")
  full_green = GreenExtractor(missing_green, bayer)
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


def build_full_model(green_model):
  bayer = Input(1, "Bayer")
  green = GreenExtractor(green_model, bayer)
  green_input = Input(1, node=green)

  # build chroma model
  diff = Sub(bayer, green_input)
  chroma_diff = Conv2D(diff, 3)
  chroma = Add(chroma_diff, green_input)
  red_blue = ChromaExtractor(chroma, bayer)
  rgb = Stack(red_blue, green)
  out = Conv2D(rgb, 3)
  out.assign_parents()
  
  return out, set((bayer.name, green_input.name))

def mini_demosaicnet():
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

