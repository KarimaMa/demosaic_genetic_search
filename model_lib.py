from demosaic_ast import *

def multires_green_model():
  # detector model
  # lowres model
  bayer = Input(1, "Bayer")
  downsample = Downsample(bayer)
  selection_f_lowres = Conv1D(downsample, 8)
  relu1 = Relu(selection_f_lowres)
  fc1 = Conv1x1(relu1, 8)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 4)
  upsample = Upsample(fc2)

  # fullres model
  bayer = Input(1, "Bayer")
  selection_f_fullres = Conv1D(bayer, 8)
  stack = Stack(upsample, selection_f_fullres)
  relu1 = Relu(stack)
  fc1 = Conv1x1(relu1, 8)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 4)
  softmax = Softmax(fc2)

  # filter model
  interp_filters = Conv1D(bayer, 4)

  mul = Mul(interp_filters, softmax)
  missing_green = SumR(mul)
  green = GreenExtractor(missing_green, bayer)
  green.assign_parents()
  return green

def multires_green_model2():
  # detector model
  # lowres model
  bayer = Input(1, "Bayer")
  downsample = Downsample(bayer)
  selection_f_lowres = Conv1D(downsample, 8)
  relu1 = Relu(selection_f_lowres)
  fc1 = Conv1x1(relu1, 8)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 4)
  softmax = Softmax(fc2)
  upsample = Upsample(softmax)

  # fullres model
  bayer = Input(1, "Bayer")
  selection_f_fullres = Conv1D(bayer, 8)
  stack = Stack(upsample, selection_f_fullres)
  relu1 = Relu(stack)
  fc1 = Conv1x1(relu1, 8)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 4)
  softmax = Softmax(fc2)

  # filter model
  interp_filters = Conv1D(bayer, 4)

  mul = Mul(interp_filters, softmax)
  missing_green = SumR(mul)
  green = GreenExtractor(missing_green, bayer)
  green.assign_parents()
  return green


def multires_green_model3():
  # detector model
  # lowres model
  bayer = Input(1, "Bayer")
  downsample = Downsample(bayer)
  selection_f_lowres = Conv1D(downsample, 8)
  relu1 = Relu(selection_f_lowres)
  fc1 = Conv1x1(relu1, 8)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 4)
  upsample = Upsample(fc2)

  # fullres model
  bayer = Input(1, "Bayer")
  selection_f_fullres = Conv1D(bayer, 8)
  relu1 = Relu(selection_f_fullres)
  fc1 = Conv1x1(relu1, 8)
  stack = Stack(upsample, fc1)
  relu2 = Relu(stack)
  fc2 = Conv1x1(relu2, 4)
  softmax = Softmax(fc2)

  # filter model
  interp_filters = Conv1D(bayer, 4)

  mul = Mul(interp_filters, softmax)
  missing_green = SumR(mul)
  green = GreenExtractor(missing_green, bayer)
  green.assign_parents()
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

