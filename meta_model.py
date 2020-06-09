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

    self.chroma_inputs = set(("Input(GreenExtractor)", "Bayer"))
    self.green_inputs = set(("Bayer"))
    self.touchup_inputs = set(("Input(GreenExtractor)", "Input(ChromaExtractor)"))
  
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

  
   
