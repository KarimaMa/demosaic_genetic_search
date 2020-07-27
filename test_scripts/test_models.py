from demosaic_ast import *

def build_linear_model():
  bayer_l = Input(1, "Bayer")
  d1filters_l = Conv1D(bayer_l, 8)
  fc1l = Conv1x1(d1filters_l, 8)
  
  bayer_r = Input(1)
  fc1r = Conv1x1(bayer_r, 8)
  add = Add(fc1r, fc1l)
  
  return add


"""
check that green model type checks
"""
def build_green_model():
  # detector model
  bayer = Input(1, "Bayer")
  d1filters_d = Conv1D(bayer, 8)
  relu1 = Relu(d1filters_d)
  fc1 = Conv1x1(relu1, 8)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 4)
  softmax = Softmax(fc2)

  # filter model
  d1filters_f = Conv1D(bayer, 4)

  mul = Mul(d1filters_f, softmax)
  out = SumR(mul)
  return out


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

