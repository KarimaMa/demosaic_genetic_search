from demosaic_ast import *

"""
Build a default model
""" 
def build_green_model():
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

def build_chroma_model(green):
  bayer = Input(1, "Bayer")
  green_input = Input(1, node=green)

  # build chroma model
  diff = Sub(bayer, green_input)
  chroma_diff = Conv2D(diff, 16)
  missing_chroma = Add(chroma_diff, green_input)

  bayer = Input(1, "Bayer")
  full_chroma = ChromaExtractor(missing_chroma, bayer)
  full_chroma.is_root = True
  full_chroma.assign_parents()
  full_chroma.compute_input_output_channels()
  return full_chroma

def build_touchup_model(green, chroma):
  green_input = Input(1, node=green)
  chroma_input = Input(1, node=chroma)
  rgb = Stack(chroma_input, green_input)
  out = Conv2D(rgb, 3)
  out.is_root = True
  out.assign_parents()
  out.compute_input_output_channels()
  return out


"""
Defines the macro structure for the demosaicking program
"""
class MetaModel():
  def __init__(self, chroma_model=None, green_model=None, touchup_model=None):
    self.chroma = chroma_model
    self.green = green_model
    self.touchup = touchup_model

    self.chroma_out_c = 3
    self.green_out_c = 1
    self.touchup_out_c = 3

    self.chroma_inputs = set(("Input(GreenExtractor)", "Input(Bayer)"))
    self.green_inputs = set(("Input(Bayer)",))
    self.touchup_inputs = set(("Input(GreenExtractor)", "Input(ChromaExtractor)"))
  
  def build_default_model(self):
    self.green = build_green_model()
    self.chroma = build_chroma_model(self.green)
    self.touchup = build_touchup_model(self.green, self.chroma)

  """
  check channel counts and inputs of chroma, green, and touchup models
  """
  def check_channel_counts(self):
    assert self.green.out_c == self.green_out_c, "Green model must have 1 output channel".format(self.green_out_c)
    assert self.chroma.out_c == self.chroma_out_c, "Chroma model must have {} output channels".format(self.chroma_out_c)
    assert self.touchup.out_c == self.touchup_out_c, "Touchup model must have {} output channels".format(self.touchup_out_c)

  def check_iputs(self):
    chroma_inputs = self.chroma.get_inputs()
    assert chroma_inputs == self.chroma_inputs, "Invalid inputs for Chroma model"
    green_inputs = self.green.get_inputs()
    assert green_inputs == self.green_inputs, "Invalid inputs for Green model"
    touchup_inputs = self.touchup.get_inputs()
    assert touchup_inputs == self.touchup_inputs, "Invalid inputs for Touchup model"


