from demosaic_ast import *
from type_check import *


def build_linear_model():
  bayer_l = Input(1)
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
  bayer_d = Input(1)
  d1filters_d = Conv1D(bayer_d, 8)
  relu1 = Relu(d1filters_d)
  fc1 = Conv1x1(relu1, 8)
  relu2 = Relu(fc1)
  fc2 = Conv1x1(relu2, 4)
  softmax = Softmax(fc2)

  # filter model
  bayer_f = Input(1)
  d1filters_f = Conv1D(bayer_f, 4)

  mul = Mul(d1filters_f, softmax)
  out = SumR(mul)
  return out


def build_full_model(green_model):
  bayer_g = Input(1)
  green = GreenExtractor(green_model, bayer_g)
  # build chroma model
  bayer = Input(1)
  diff = Sub(bayer, green)
  chroma_diff = Conv2D(diff, 3)
  chroma = Add(chroma_diff, green)

  bayer_rb = Input(1)
  red_blue = ChromaExtractor(chroma, bayer_rb)
  
  rgb = Stack(red_blue, green)
  out = Conv2D(rgb, 3)
  return out


green_model = build_green_model()

try:
  out_c = check_channel_count(green_model) 
except AssertionError:
  print("Failed channel count")
else:
  print("Model output channel count {}".format(out_c))

# try to change output channel counts
# this should fail since the root node is a sum reduce
fix_channel_count(green_model, 12)
print("trying to change channel count of green model")
try:
  out_c = check_channel_count(green_model) 
except AssertionError:
  print("Failed channel count")
else:
  print("Model output channel count {}".format(out_c))
  
green_model.dump("")

try:
  check_linear_types(green_model)
except AssertionError:
  print("failed linear check")
else:
  print("passed linear check")

# build a model that should fail the linear check
linear_model = build_linear_model()

try:
  check_linear_types(linear_model)
except AssertionError:
  print("failed linear check")
else:
  print("passed linear check")


green_model = build_green_model()
full_model = build_full_model(green_model)
full_model.dump("")
try:
  out_c = check_channel_count(full_model) 
except AssertionError:
  print("Failed channel count")
else:
  print("Model output channel count {}".format(out_c))
  
try:
  check_linear_types(full_model)
except AssertionError:
  print("failed linear check")
else:
  print("passed linear check")

