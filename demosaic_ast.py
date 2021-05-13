f"""
TODO:
  ADD GREEN AND CHROMA EXTRACTORS
"""
from abc import ABC, abstractmethod
from tree import Node, has_loop, hash_combine
import copy
import sys
import pickle
from orderedset import OrderedSet
from inspect import signature

# from a github gist by victorlei
def extclass(cls):
  return lambda f: (setattr(cls,f.__name__,f) or f)
""" ----------------------------------------------"""

class Binop(ABC):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class BinopIII(Binop):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class BinopIJK(Binop):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class BinopIcJcKc(Binop):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"
  @property
  @abstractmethod
  def Ic(self):
      pass
  @property
  @abstractmethod
  def Jc(self):
    pass
  @property
  @abstractmethod
  def Kc(self):
    pass

class Ternary(ABC):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class TernaryHcIcJcKc(Ternary):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"
  @property
  @abstractmethod
  def Hc(self):
    pass
  @property
  @abstractmethod
  def Ic(self):
    pass
  @property
  @abstractmethod
  def Jc(self):
    pass
  @property
  @abstractmethod
  def Kc(self):
    pass

class Unop(ABC):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class UnopIcJc(Unop):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class UnopII(Unop):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class UnopIJ(Unop):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class UnopIIdiv(Unop):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class UnopIJFixed(Unop):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class Const(ABC):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class NonLinear(ABC):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class Linear(ABC):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class Special(ABC):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

"""----------------------------------------------------------------------"""

class Upsample(Unop, Special, Node):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class Downsample(Unop, Special, Node):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"


"""----------------------------------------------------------------------"""

class Input(Const, Special, Node):
  def __init__(self, out_c, resolution=None, name=None, node=None, no_grad=None, green_model_id=None):
    if node:
      if name is None:
        name = node.name

      self.node = node
      assert(out_c == node.out_c), "output channels of node to input doesn't match given out_c"
    
    if not green_model_id is None:
      self.green_model_id = green_model_id

    if not no_grad is None:
      self.no_grad = no_grad

    Node.__init__(self, "Input({})".format(name), 0)
    self.resolution = resolution
    self.in_c = out_c
    self.out_c = out_c 

class Add(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "Add"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

class Sub(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "Sub"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

class Mul(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "Mul"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

class LogSub(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "LogSub"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

class AddExp(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "AddExp"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

class Stack(BinopIJK, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "Stack"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.out_c = None
    self.in_c = None 
    self.resolution = resolution
    

# Takes FullGreenQuad predction, RedBlueQuad from Bayer, and 
# 6 channel chroma quad prection: R@Gr, B@R, R@B, R@Gb, B@Gr, B@Gb
# spits out full RGB Image at full resolution
class RGBExtractor(TernaryHcIcJcKc, Special, Node):
  def __init__(self, child1, child2, child3, resolution=None, name=None):
    if name is None:
      name = "RGBExtractor"
    Node.__init__(self, name, 3)
    self.child1 = child1
    self.child2 = child2
    self.child3 = child3
    self.in_c = (4, 2, 6) # fullgreen, redbluebayer, missing chroma
    self.out_c = 3
    self.resolution = resolution

  def Hc(self):
    return 4 # fullgreen
  def Ic(self):
    return 2 # redblue bayer
  def Jc(self):
    return 6 # missing chroma
  def Kc(self): 
    return 3 # full rgb image


class XRGBExtractor(TernaryHcIcJcKc, Special, Node):
  def __init__(self, child1, child2, child3, resolution=None, name=None):
    if name is None:
      name = "XRGBExtractor"
    Node.__init__(self, name, 3)
    self.child1 = child1
    self.child2 = child2
    self.child3 = child3
    self.in_c = (36, 36, 56) # fullgreen, redbluebayer, missing chroma
    self.out_c = 3
    self.resolution = resolution

  def Hc(self):
    return 36 # fullgreen
  def Ic(self):
    return 36 #  xtrans
  def Jc(self):
    return 56 # missing chroma
  def Kc(self): 
    return 3 # full rgb image

class XFlatRGBExtractor(TernaryHcIcJcKc, Special, Node):
  def __init__(self, child1, child2, child3, resolution=None, name=None):
    if name is None:
      name = "XFlatRGBExtractor"
    Node.__init__(self, name, 3)
    self.child1 = child1
    self.child2 = child2
    self.child3 = child3
    self.in_c = (1, 3, 2) # flat green, xtrans 3 channel, chroma pred 
    self.out_c = 3
    self.resolution = resolution

  def Hc(self):
    return 1 # fullgreen
  def Ic(self):
    return 3 #  xtrans
  def Jc(self):
    return 2 # missing chroma
  def Kc(self): 
    return 3 # full rgb image


class RGBSuperResExtractor(BinopIcJcKc, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "RGBSuperResExtractor"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.in_c = (1, 2) # green, red and blue
    self.out_c = 3
    self.resolution = resolution

  def Ic(self):
    return 1 # superres green
  def Jc(self):
    return 2 # missing chroma
  def Kc(self): 
    return 3 # full rgb image



# Takes bayer quad, 8 channel predction
# spits out full RGB Image at full resolution
class RGB8ChanExtractor(BinopIcJcKc, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "RGB8ChanExtractor"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.in_c = (4, 8) # bayer quad, 8 channel prediction
    self.out_c = 3
    self.resolution = resolution

  def Ic(self):
    return 4 
  def Jc(self):
    return 8 
  def Kc(self): 
    return 3 # full rgb image


# Takes flat 3 channel mosaic 8 channel predction
# spits out full RGB Image at full resolution
class FlatRGB8ChanExtractor(BinopIcJcKc, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "FlatRGB8ChanExtractor"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.in_c = (3, 8) # bayer, 8 channel prediction
    self.out_c = 3
    self.resolution = resolution

  def Ic(self):
    return 3
  def Jc(self):
    return 8 
  def Kc(self): 
    return 3 # full rgb image


"""
Takes BayerQuad and 2 channel green prediction 
Spits out full Green image at full resolution  
"""
class GreenExtractor(BinopIcJcKc, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "GreenExtractor"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.in_c = (4, 2) # bayer, missing green
    self.out_c = 1 # output G channel
    self.resolution = resolution
  
  def Ic(self):
    return 4
  def Jc(self):
    return 2
  def Kc(self):
    return 1


"""
Dummy border op wrapper for superres task 
"""
class SGreenExtractor(UnopIcJc, Special, Node):
  def __init__(self, child, resolution=None, name=None):
    if name is None:
      name = "SGreenExtractor"
    Node.__init__(self, name, 1)
    self.child = child
    self.in_c = 1
    self.out_c = 1 # output G channel
    self.resolution = resolution
  
  def Ic(self):
    return 1
  def Jc(self):
    return 1

"""
Xtrans extractor for green channel 
16 missing green values per 36 mosaic values
"""
class XGreenExtractor(BinopIcJcKc, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "XGreenExtractor"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.in_c = (36, 16) # bayer, missing green
    self.out_c = 1
    self.resolution = resolution
 
  def Ic(self):
    return 36
  def Jc(self):
    return 16
  def Kc(self):
    return 1


class XFlatGreenExtractor(BinopIcJcKc, Special, Node):
  def __init__(self, lchild, rchild, resolution=None, name=None):
    if name is None:
      name = "XFlatGreenExtractor"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.in_c = (3, 1) # bayer, missing green
    self.out_c = 1
    self.resolution = resolution
  
  def Ic(self):
    return 3
  def Jc(self):
    return 1
  def Kc(self):
    return 1


class Flat2Quad(UnopIcJc, Special, Node):
  def __init__(self, child, resolution=None, name=None):
    if name is None:
      name = "Flat2Quad"
    Node.__init__(self, name, 1)
    self.child = child
    self.in_c = 1
    self.out_c = 4 
    self.resolution = resolution
  
  def Ic(self):
    return 1
  def Jc(self):
    return 4

class GreenRBExtractor(UnopIcJc, Special, Node):
  def __init__(self, child, resolution=None, name=None):
    if name is None:
      name = "GreenRBExtractorOp"
    Node.__init__(self, name, 1)
    self.child = child
    self.in_c = 1
    self.out_c = 2
    self.resolution = resolution
  
  def Ic(self):
    return 1
  def Jc(self):
    return 2

"""
Takes flat green prediction returns 6x6 packed 
green values at xtrans R and B locations
"""
class XGreenRBExtractor(UnopIcJc, Special, Node):
  def __init__(self, child, resolution=None, name=None):
    if name is None:
      name = "XGreenRBExtractorOp"
    Node.__init__(self, name, 1)
    self.child = child
    self.in_c = 1
    self.out_c = 16
    self.resolution = resolution
  
  def Ic(self):
    return 1
  def Jc(self):
    return 16
 
 
class Softmax(UnopII, NonLinear, Node):
  def __init__(self, child, resolution=None, name=None):
    if name is None:
      name = "Softmax"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

class Relu(UnopII, NonLinear, Node):
  def __init__(self, child, resolution=None, name=None):
    if name is None:
      name = "Relu"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

class Log(UnopII, NonLinear, Node):
  def __init__(self, child, resolution=None, name=None):
    if name is None:
      name = "Log"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

class Exp(UnopII, NonLinear, Node):
  def __init__(self, child, resolution=None, name=None):
    if name is None:
      name = "Exp"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

class LearnedDownsample(UnopIJ, Downsample):
  def __init__(self, child, out_c:int, factor=None, groups=1, resolution=None, name=None):
    if name is None:
      name = "LearnedDownsample"
    Node.__init__(self, name, 1)
    self.child = child
    self.factor = factor
    self.out_c = out_c
    self.in_c = None
    self.groups = groups
    self.resolution = resolution


# class PeriodicConv(UnopIJFixed, Special, Node):
#   def __init__(self, child, out_c:int, period=None, kwidth=3, resolution=None, name=None):
#     if name is None:
#       name = "PeriodicConv"
#     Node.__init__(self, name, 1)
#     self.child = child
#     self.period = period
#     self.kwidth = kwidth
#     self.out_c = out_c
#     self.in_c = None  
#     self.resolution = resolution


class Pack(UnopIJFixed, Downsample):
  def __init__(self, child, factor=None, resolution=None, name=None):
    if name is None:
      name = "Pack"
    Node.__init__(self, name, 1)
    self.child = child
    self.factor = factor
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

# class LearnedPack(UnopIJFixed, Special, Node):
#   def __init__(self, child, factor=None, resolution=None, name=None):
#     if name is None:
#       name = "LearnedPack"
#     Node.__init__(self, name, 1)
#     self.child = child
#     self.factor = factor
#     self.out_c = None
#     self.in_c = None
#     self.resolution = resolution

class Unpack(UnopIJFixed, Upsample):
  def __init__(self, child, factor=None, resolution=None, name=None):
    if name is None:
      name = "Unpack"
    Node.__init__(self, name, 1)
    self.child = child
    self.factor = factor
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

"""
Fixed Upsample op, no learned parameters
"""
class BilinearUpsample(UnopII, Upsample):
  def __init__(self, child, factor=None, resolution=None, name=None):
    if name is None:
      name = "BilinearUpsample"
    Node.__init__(self, name, 1)
    self.child = child
    self.factor = factor
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

"""
Scale factor cannot be mutated, thus the output channels is 
a fixed function of the input channels, specifically, 
out_c = in_c / factor^2
Grouping can be changed since this is implemented with a transposed conv
"""
class LearnedUpsample(UnopIJ, Upsample):
  def __init__(self, child, out_c: int, factor=None, groups=1, resolution=None, name=None):
    if name is None:
      name = "LearnedUpsample"
    Node.__init__(self, name, 1)
    self.child = child
    self.factor = factor
    self.groups = groups 
    self.out_c = out_c
    self.in_c = None
    self.resolution = resolution

class FastUpsample(UnopII, Special, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "FastUpsample"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = None
    self.in_c = None
    self.resolution = resolution

class Conv1x1(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int, groups=1, resolution=None, name=None):
    if name is None:
      name = "Conv1x1"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = out_c
    self.in_c = None
    self.groups = groups
    self.resolution = resolution

class Conv1D(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int, groups=1, kwidth=3, resolution=None, name=None):
    if name is None:
      name = "Conv1D"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = out_c
    self.kwidth = kwidth
    self.in_c = None
    self.groups = groups
    self.resolution = resolution

class Conv2D(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int, groups=1, kwidth=3, resolution=None, name=None):
    if name is None:
      name = "Conv2D"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = out_c
    self.kwidth = kwidth
    self.in_c = None
    self.groups = groups
    self.resolution = resolution

class InterleavedSum(UnopIIdiv, Special, Node):
  def __init__(self, child, out_c: int, resolution=None, name=None):
    if name is None:
      name = "InterleavedSum"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = out_c
    self.in_c = None
    self.resolution = resolution

class GroupedSum(UnopIIdiv, Special, Node):
  def __init__(self, child, out_c: int, resolution=None, name=None):
    if name is None:
      name = "GroupedSum"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = out_c
    self.in_c = None
    self.resolution = resolution


@extclass(Input)
def compute_input_output_channels(self):
  return self.in_c, self.out_c

@extclass(Add)
def compute_input_output_channels(self):
  leftchild_in_c, leftchild_out_c = self.lchild.compute_input_output_channels()
  rightchild_in_c, rightchild_out_c = self.rchild.compute_input_output_channels()
  self.in_c = (leftchild_out_c, rightchild_out_c)
  self.out_c = max(leftchild_out_c, rightchild_out_c)
  return self.in_c, self.out_c

@extclass(Sub)
def compute_input_output_channels(self):
  leftchild_in_c, leftchild_out_c = self.lchild.compute_input_output_channels()
  rightchild_in_c, rightchild_out_c = self.rchild.compute_input_output_channels()
  self.in_c = (leftchild_out_c, rightchild_out_c)
  self.out_c = max(leftchild_out_c, rightchild_out_c)
  return self.in_c, self.out_c

@extclass(Mul)
def compute_input_output_channels(self):
  leftchild_in_c, leftchild_out_c = self.lchild.compute_input_output_channels()
  rightchild_in_c, rightchild_out_c = self.rchild.compute_input_output_channels()
  self.in_c = (leftchild_out_c, rightchild_out_c)
  self.out_c = max(leftchild_out_c, rightchild_out_c)
  return self.in_c, self.out_c

@extclass(LogSub)
def compute_input_output_channels(self):
  leftchild_in_c, leftchild_out_c = self.lchild.compute_input_output_channels()
  rightchild_in_c, rightchild_out_c = self.rchild.compute_input_output_channels()
  self.in_c = (leftchild_out_c, rightchild_out_c)
  self.out_c = max(leftchild_out_c, rightchild_out_c)
  return self.in_c, self.out_c

@extclass(AddExp)
def compute_input_output_channels(self):
  leftchild_in_c, leftchild_out_c = self.lchild.compute_input_output_channels()
  rightchild_in_c, rightchild_out_c = self.rchild.compute_input_output_channels()
  self.in_c = (leftchild_out_c, rightchild_out_c)
  self.out_c = max(leftchild_out_c, rightchild_out_c)
  return self.in_c, self.out_c

@extclass(Stack)
def compute_input_output_channels(self):
  _, lout_c = self.lchild.compute_input_output_channels()
  _, rout_c = self.rchild.compute_input_output_channels()
  self.in_c = (lout_c, rout_c)
  self.out_c = lout_c + rout_c
  return self.in_c, self.out_c

@extclass(RGBExtractor)
def compute_input_output_channels(self):
  self.child1.compute_input_output_channels()
  self.child2.compute_input_output_channels()
  self.child3.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(XRGBExtractor)
def compute_input_output_channels(self):
  self.child1.compute_input_output_channels()
  self.child2.compute_input_output_channels()
  self.child3.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(XFlatRGBExtractor)
def compute_input_output_channels(self):
  self.child1.compute_input_output_channels()
  self.child2.compute_input_output_channels()
  self.child3.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(RGBSuperResExtractor)
def compute_input_output_channels(self):
  self.lchild.compute_input_output_channels()
  self.rchild.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(RGB8ChanExtractor)
def compute_input_output_channels(self):
  self.lchild.compute_input_output_channels()
  self.rchild.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(FlatRGB8ChanExtractor)
def compute_input_output_channels(self):
  self.lchild.compute_input_output_channels()
  self.rchild.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(GreenExtractor)
def compute_input_output_channels(self):
  self.lchild.compute_input_output_channels()
  self.rchild.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(SGreenExtractor)
def compute_input_output_channels(self):
  self.child.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(XGreenExtractor)
def compute_input_output_channels(self):
  self.lchild.compute_input_output_channels()
  self.rchild.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(XFlatGreenExtractor)
def compute_input_output_channels(self):
  self.lchild.compute_input_output_channels()
  self.rchild.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(Flat2Quad)
def compute_input_output_channels(self):
  self.child.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(GreenRBExtractor)
def compute_input_output_channels(self):
  self.child.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(XGreenRBExtractor)
def compute_input_output_channels(self):
  self.child.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(Softmax)
def compute_input_output_channels(self):
  _, lout_c = self.child.compute_input_output_channels()
  self.in_c = lout_c
  self.out_c = lout_c
  return self.in_c, self.out_c
    
@extclass(Relu)
def compute_input_output_channels(self):
  _, lout_c = self.child.compute_input_output_channels()
  self.in_c = lout_c
  self.out_c = lout_c
  return self.in_c, self.out_c

@extclass(Log)
def compute_input_output_channels(self):
  _, lout_c = self.child.compute_input_output_channels()
  self.in_c = lout_c
  self.out_c = lout_c
  return self.in_c, self.out_c

@extclass(Exp)
def compute_input_output_channels(self):
  _, lout_c = self.child.compute_input_output_channels()
  self.in_c = lout_c
  self.out_c = lout_c
  return self.in_c, self.out_c

@extclass(LearnedDownsample)
def compute_input_output_channels(self):
  child_in_c, child_out_c = self.child.compute_input_output_channels()
  self.in_c = child_out_c
  return self.in_c, self.out_c

# @extclass(PeriodicConv)
# def compute_input_output_channels(self):
#   _, child_out_c = self.child.compute_input_output_channels()
#   self.in_c = child_out_c
#   return self.in_c, self.out_c

@extclass(Pack)
def compute_input_output_channels(self):
  _, lout_c = self.child.compute_input_output_channels()
  self.in_c = lout_c
  self.out_c = int(lout_c * self.factor**2)
  return self.in_c, self.out_c

# @extclass(LearnedPack)
# def compute_input_output_channels(self):
#   child_in_c, child_out_c = self.child.compute_input_output_channels()
#   self.in_c = child_out_c
#   self.out_c = self.in_c * self.factor**2
#   return self.in_c, self.out_c

@extclass(Unpack)
def compute_input_output_channels(self):
  _, lout_c = self.child.compute_input_output_channels()
  self.in_c = lout_c
  self.out_c = int(lout_c // self.factor**2)
  return self.in_c, self.out_c

@extclass(BilinearUpsample)
def compute_input_output_channels(self):
  _, lout_c = self.child.compute_input_output_channels()
  self.in_c = lout_c
  self.out_c = lout_c
  return self.in_c, self.out_c

@extclass(LearnedUpsample)
def compute_input_output_channels(self):
  child_in_c, child_out_c = self.child.compute_input_output_channels()
  self.in_c = child_out_c
  return self.in_c, self.out_c

@extclass(FastUpsample)
def compute_input_output_channels(self):
  _, lout_c = self.child.compute_input_output_channels()
  self.in_c = lout_c
  self.out_c = lout_c
  return self.in_c, self.out_c

@extclass(Conv1x1)
def compute_input_output_channels(self):
  child_in_c, child_out_c = self.child.compute_input_output_channels()
  self.in_c = child_out_c
  return self.in_c, self.out_c

@extclass(Conv1D)
def compute_input_output_channels(self):
  child_in_c, child_out_c = self.child.compute_input_output_channels()
  self.in_c = child_out_c
  return self.in_c, self.out_c

@extclass(Conv2D)
def compute_input_output_channels(self):
  child_in_c, child_out_c = self.child.compute_input_output_channels()
  self.in_c = child_out_c
  return self.in_c, self.out_c

@extclass(InterleavedSum)
def compute_input_output_channels(self):
  _, child_out_c = self.child.compute_input_output_channels()
  self.in_c = child_out_c
  assert self.in_c % self.out_c == 0, f"InterleavedSum {self.dump()} must have input channels {self.in_c} divisible by output channels {self.out_c}"
  return self.in_c, self.out_c

@extclass(GroupedSum)
def compute_input_output_channels(self):
  _, child_out_c = self.child.compute_input_output_channels()
  self.in_c = child_out_c
  assert self.in_c % self.out_c == 0, f"GroupedSum {self.dump()} must have input channels {self.in_c} divisible by output channels {self.out_c}"
  return self.in_c, self.out_c

"""
Converts AST structure to array format
"""
@extclass(Node)
def structure_to_array(self):
  preorder = self.preorder()
  array = []
  for i, n in enumerate(preorder):
    node_info = {"type": instance_to_classnamme(n), "in_c": n.in_c, "out_c": n.out_c, "name": n.name}
    # find all preorder ids of nodes in partner set
    if hasattr(n, 'partner_set'):
      partner_ids = []
    
      for (partner_node, node_id) in n.partner_set:
        found = False
        for j in range(len(preorder)):
          if partner_node is preorder[j]:
            partner_ids += [j]
            found = True 
            break

      node_info["partner_set"] = partner_ids

    parents = []
    seen_parents = OrderedSet()
    if type(n.parent) is tuple:
      for node_parent in n.parent:
        for j in range(0, len(preorder)):
          if id(preorder[j]) == id(node_parent) and not id(preorder[j]) in seen_parents:
            parents += [j]
            seen_parents.add(id(preorder[j]))
    else:
      for j in range(0, i):
        if id(preorder[j]) == id(n.parent) and not id(preorder[j]) in seen_parents:
          parents += [j]
          seen_parents.add(id(preorder[j]))

    node_info["parent"] = parents
    if n.num_children == 3:
      child1_id = None
      child2_id = None
      child3_id = None
      # NOTE that with DAG structures it is now possible for
      # a node to have children use the same node 
      for j in range(0, len(preorder)):
        if id(preorder[j]) == id(n.child1):
          child1_id = j
        if id(preorder[j]) == id(n.child2):
          child2_id = j
        if id(preorder[j]) == id(n.child3):
          child3_id = j
        if not any([c is None for c in [child1_id, child2_id, child3_id]]):
          break
      node_info["children"] = [child1_id, child2_id, child3_id]

    elif n.num_children == 2:
      lchild_id = None
      rchild_id = None
      for j in range(0, len(preorder)):
        if id(preorder[j]) == id(n.lchild):
          lchild_id = j
        if id(preorder[j]) == id(n.rchild):
          rchild_id = j
        if not lchild_id is None and not rchild_id is None:
          break
      node_info["children"] = [lchild_id, rchild_id]

    elif n.num_children == 1:
      child_id = None
      for j in range(0, len(preorder)):
        if id(preorder[j]) == id(n.child):
          child_id = j
          break
      node_info["children"] = [child_id]
  
    if hasattr(n, 'name'):
      node_info["name"] = n.name.replace("Input(",'').rstrip(")")

    if hasattr(n, 'kwidth'):
      node_info["kwidth"] = n.kwidth

    if hasattr(n, 'groups'):
      node_info["groups"] = n.groups

    if hasattr(n, 'factor'):
      node_info['factor'] = n.factor

    if hasattr(n, 'resolution'):
      node_info['resolution'] = n.resolution

    if hasattr(n, 'node'):
      if hasattr(n, 'green_model_id'):
        node_info['green_model_id'] = n.green_model_id # only has meaning given the current search run's choice of green models
      # store info needed to reconstruct model referenced by the input node
      node_info['node_ast_info'] = n.node.structure_to_array()

    if hasattr(n, "no_grad"):
      node_info['no_grad'] = n.no_grad

    array.append(node_info)
  return array

"""
Saves the AST into a file
"""
@extclass(Node)
def save_ast(self, filename):
  tree_data = self.structure_to_array()  
  with open(filename, "wb") as f:
    pickle.dump(tree_data, f)
    

def build_tree_from_data(node_id, preorder_nodes, shared_children=None, shared_input_models=None):
  if shared_children is None:
    shared_children = {}
  if shared_input_models is None:
    shared_input_models = {}

  node_info = preorder_nodes[node_id]
  node_type = node_info["type"]
  node_class = str_to_class(node_type)
  node_name = node_info["name"]
  if "children" in node_info:
    children_ids = node_info["children"]
    children_info = [preorder_nodes[children_ids[i]] for i in range(len(children_ids))]
    child_nodes = []
    for i, child_info in enumerate(children_info):
      if len(child_info["parent"]) > 1:
        if children_ids[i] in shared_children:
          child_node = shared_children[children_ids[i]]
        else:
          child_node = build_tree_from_data(children_ids[i], preorder_nodes, shared_children, shared_input_models)
          shared_children[children_ids[i]] = child_node
      else:
        child_node = build_tree_from_data(children_ids[i], preorder_nodes, shared_children, shared_input_models)
      child_nodes.append(child_node)

    extra_kwargs = {}
    if "kwidth" in node_info:
      extra_kwargs["kwidth"] = node_info["kwidth"]
    if "groups" in node_info:
      extra_kwargs["groups"] = node_info["groups"]
    if "factor" in node_info:
      extra_kwargs["factor"] = node_info["factor"]
    if "out_c" in node_info and "out_c" in signature(node_class).parameters: #(issubclass(node_class, UnopIJ) or issubclass(node_class, UnopIIdiv)):
      extra_kwargs["out_c"] = node_info["out_c"]
    if "resolution" in node_info:
      extra_kwargs["resolution"] = node_info["resolution"]

    if len(extra_kwargs) > 0:
      new_node = node_class(*child_nodes, name=node_name, **extra_kwargs)
    else:
      new_node = node_class(*child_nodes, name=node_name)

  else: # is input node
    extra_kwargs = {}
    # made the choice to not save the node ast if input node is a sub model
    # just save the model id and assume upon reloading this model id is understood
    if "node_ast_info" in node_info:
      if "green_model_id" in node_info: # don't build the green model here, let search code insert it later
        extra_kwargs["green_model_id"] = node_info["green_model_id"]
      else: # Input op references a model that isn't a green model, build it here
        if node_name in shared_input_models:
          input_ast = shared_input_models[node_name]
        else:
          input_ast = build_tree_from_data(0, node_info["node_ast_info"], shared_input_models=shared_input_models)
          input_ast.compute_input_output_channels()
          shared_input_models[node_name] = input_ast
        extra_kwargs["node"] = input_ast
      if "no_grad" in node_info:
        extra_kwargs["no_grad"] = node_info["no_grad"]
    if "resolution" in node_info:
      extra_kwargs["resolution"] = node_info["resolution"]

    if len(extra_kwargs) > 0:
      new_node = node_class(node_info["in_c"], name=node_name, **extra_kwargs)
    else:
      new_node = node_class(node_info["in_c"], name=node_name)
  
  return new_node


def link_partners_in_reconstructed_tree(tree, preorder_node_info):
  preorder = tree.preorder()
  for i, node_info in enumerate(preorder_node_info):
    if "partner_set" in node_info:
      partner_ids = node_info["partner_set"] # list of preorder ids of partners
      partner_set = OrderedSet([ (preorder[pid], id(preorder[pid])) for pid in partner_ids])
      preorder[i].partner_set = partner_set

def load_ast(filename):
  with open(filename, "rb") as f:
    tree_data = pickle.load(f)
  tree = build_tree_from_data(0, tree_data)
  tree.compute_input_output_channels()
  link_partners_in_reconstructed_tree(tree, tree_data)
  tree.assign_parents()

  return tree

def instance_to_classnamme(o):
  return o.__class__.__name__

def str_to_class(str):
  return getattr(sys.modules[__name__], str)


"""
given a full model, finds the green model id used by the full model
and asserts that only one type of green model is used
"""
def get_green_model_id(full_model_ast):
  green_model_id = None

  nodes = full_model_ast.preorder()
  for n in nodes:
    if type(n) is Input:
      if n.name == "Input(GreenExtractor)":
        if green_model_id is None:  
          green_model_id = n.green_model_id
        else:
          assert green_model_id == n.green_model_id, "chroma DAG must use only one type of green model"
      elif hasattr(n, 'node'):
        submodel_green_model_id = get_green_model_id(n.node)
        if green_model_id is None:
          green_model_id = submodel_green_model_id
        else:
          assert submodel_green_model_id == green_model_id, "chroma DAG must use only one type of green model"
  return green_model_id


"""
given a full model, sets the green model id used by the full model
to the given green model id
"""
def set_green_model_id(full_model_ast, green_model_id):
  nodes = full_model_ast.preorder()
  for n in nodes:
    if type(n) is Input:
      if n.name == "Input(GreenExtractor)":
        n.green_model_id = green_model_id
      elif hasattr(n, 'node'):
        set_green_model_id(n.node, green_model_id)


"""
returns an integer that represents high level structural 
information of the given tree
"""
def structural_hash(tree):
  nodes = tree.preorder()
  h = 0
  for i,n in enumerate(nodes):
    if isinstance(n, Binop):
      h += 1 << i

  op_list = list(all_ops)

  ops_used = OrderedSet([type(n) for n in nodes])
  op_coverage = 0
  for i,o in enumerate(op_list):
    if o in ops_used:
      op_coverage += 1 << i

  sh = str(h) + str(op_coverage) 

  green_model_id = get_green_model_id(tree)
  if green_model_id:
    sh += str(green_model_id)

  return sh
  
"""
returns a deep copy of the subtree at root but does NOT
make a deep copy of ancestors. and deletes original references 
to the parent tree
"""
def copy_subtree_helper(root, seen_children=None):
  if seen_children is None:
    seen_children = {}

  if id(root) in seen_children:
    return seen_children[id(root)]

  new_root = copy.copy(root)
  new_root.parent = None

  children_copies = []

  def copy_children():
    for c in children:
      child_copy = copy_subtree_helper(c, seen_children)
      if child_copy.parent:
        if type(child_copy.parent) is tuple:
          child_copy.parent = child_copy.parent + tuple([new_root])
        else:
          child_copy.parent = (child_copy.parent, new_root) 
      else:
        child_copy.parent = new_root
      children_copies.append(child_copy)

  if new_root.num_children == 3:
    children = [root.child1, root.child2, root.child3]
    copy_children()

    new_root.child1 = children_copies[0]
    new_root.child2 = children_copies[1]
    new_root.child3 = children_copies[2]

  elif new_root.num_children == 2:
    children = [root.lchild, root.rchild]
    copy_children()

    new_root.lchild = children_copies[0]
    new_root.rchild = children_copies[1]
    
  elif new_root.num_children == 1:
    children = [root.child]
    copy_children()

    new_root.child = children_copies[0]

  if hasattr(new_root, 'node'):
    new_root.node = copy_subtree(new_root.node)

  seen_children[id(root)] = new_root
  return new_root


def redirect_to_new_partners(old_tree, new_tree):
  old_tree_preorder = old_tree.preorder()
  new_tree_preorder = new_tree.preorder()
  old_tree_ids = [id(n) for n in old_tree_preorder]

  done = OrderedSet()
  for new_node in new_tree_preorder:
    if id(new_node) in done:
      continue
    done.add(id(new_node))
    if hasattr(new_node, "partner_set"):
      found = False
      new_node_partner_set = OrderedSet()
      for p in new_node.partner_set:
        if not p[1] in old_tree_ids: # partner of new node is not within the copied subtree -it's in the shared parent tree
          pass
          #print(f"{new_node} partner not in old tree ids")

          # new_node_partner = p

          # # remove all connections between partner node and old nodes 
          # partners_of_new_node_partner = OrderedSet()
          # for item in new_node_partner[0].partner_set: 
          #   # nodes may have been modified since insertion into partner set
          #   # so we need to reproduce the set by rehashing the nodes
          #   partners_of_new_node_partner.add(item)

          # list_partners_of_new_node_partner = list(partners_of_new_node_partner)
          # for old_p in list_partners_of_new_node_partner:
          #   for old_node in old_tree_preorder:
          #     if old_p[0] is old_node:
          #       partners_of_new_node_partner.remove(old_p)
          #       break
          # new_node_partner[0].partner_set = partners_of_new_node_partner
          # new_node_partner[0].partner_set.add((new_node, id(new_node)))
          # found = True
        else: # partner of new node is within the copied subtree
          for i, old_node in enumerate(old_tree_preorder):
            if old_node is p[0]:
              new_node_partner = (new_tree_preorder[i], id(new_tree_preorder[i]))
              found = True

        if not found:
          #logger.debug(f"could not find partner {p[0].name} for node {new_node.name} within copied subtree")
          pass
        else:
          new_node_partner_set.add(new_node_partner)

      new_node.partner_set = new_node_partner_set

def copy_subtree(old_tree):
  new_tree = copy_subtree_helper(old_tree)
  redirect_to_new_partners(old_tree, new_tree)
  return new_tree




"""
def find_old_partners(tree, partner_dic={}):
  if hasattr(tree, "partner"):
    partner_dic[id(tree)] = id(tree.partner)
  if tree.num_children == 2:
    find_old_partners(tree.lchild, partner_dic)
    find_old_partners(tree.rchild, partner_dic)
  elif tree.num_children == 1:
    find_old_partners(tree.child, partner_dic)

def assign_new_partners_from_old(newtree, old_partner_dic, new_partner_dic={}):
  if hasattr(newtree, "partner"):
    old_partner_id = id(newtree.partner)
    if not old_partner_id in new_partner_dic:
      oldtree_id = old_partner_dic[old_partner_id]
      new_partner_dic[oldtree_id] = newtree
    else:
      newtree.partner = new_partner_dic[old_partner_id]
      newtree.partner.partner = newtree
  if newtree.num_children == 2:
    assign_new_partners_from_old(newtree.lchild, old_partner_dic, new_partner_dic)
    assign_new_partners_from_old(newtree.rchild, old_partner_dic, new_partner_dic)
  elif newtree.num_children == 1:
    assign_new_partners_from_old(newtree.child, old_partner_dic, new_partner_dic)

def reassign_partners_in_copy(oldtree, newtree, partner_dic={}):
  if hasattr(oldtree, "partner"):
    if id(root.partner) in partner_dic:
      new_root.partner = partner_dic[id(root.partner)]["new_partner1"]
      partner_dic[id(root.partner)]["new_partner2"] = new_root
      
      if new_root.num_children == 2:
        lcopy = copy.deepcopy(new_root.lchild, partner_dic)
        rcopy = copy.deepcopy(new_root.rchild, partner_dic)
        new_root.lchild = lcopy
        new_root.rchild = rcopy
        lcopy.parent = new_root
        rcopy.parent = new_root
      elif new_root.num_children == 1:
        child_copy = copy.deepcopy(new_root.child, partner_dic)
        new_root.child = child_copy
        child_copy.parent = new_root
    else: # waiting for partner lower in tree to add itself to dictionary
      partner_dic[id(root)] = {"new_partner1": new_root}

      if new_root.num_children == 2:
        lcopy = copy.deepcopy(new_root.lchild, partner_dic)
        rcopy = copy.deepcopy(new_root.rchild, partner_dic)
        new_root.lchild = lcopy
        new_root.rchild = rcopy
        lcopy.parent = new_root
        rcopy.parent = new_root
      elif new_root.num_children == 1:
        child_copy = copy.deepcopy(new_root.child, partner_dic)
        new_root.child = child_copy
        child_copy.parent = new_root

      new_root.partner = partner_dic[id(root)]["new_partner2"]

  new_root.parent = None
  return new_root
"""


"""
Returns all ancestors up to the closest ancestor of the given node 
that inherits from any class in the set OpClasses.
If we reach a node with multiple parents before we find the ancestor,
stop and return the list of nodes 

Assumes assign_parents() has already been called on the full tree.
Returns all the nodes between the found ancestor and the node 
NOT including the ancestor that is a member of OpClasses or has a tuple of parents

"""
def find_closest_ancestor(node, OpClasses):
  nodes = []
  while (not node.parent is None):
    parent_type = type(node.parent)
    if parent_type is tuple or any([issubclass(parent_type, oc) for oc in OpClasses]):
      break
    nodes = [node] + nodes
    node = node.parent
  return nodes

"""
Returns refernces to the closest parents belonging to the 
given opclasses.
"""
def find_closest_parents(node, OpClasses):
  if any([issubclass(type(node), oc) for oc in OpClasses]):
      return OrderedSet( [ (node, id(node)) ] )
  if node.parent is None:
    return OrderedSet()
  parent_type = type(node.parent)
  if parent_type is tuple:
    found_set1 = find_closest_parents(node.parent[0], OpClasses)
    found_set2 = find_closest_parents(node.parent[1], OpClasses)
    return found_set1.union(found_set2)
  return find_closest_parents(node.parent, OpClasses)
 

"""
Returns a reference to the closest child belonging to the 
given opclasses. If we encounter a ternary or binary op before reaching
such a child, return the closest found in each branch
"""
def find_closest_children(node, OpClasses):
  if any([issubclass(type(node), oc) for oc in OpClasses]):
    return OrderedSet( [(node, id(node))] )
  if node.num_children == 3:
    found1 = find_closest_children(node.child1, OpClasses)
    found2 = find_closest_children(node.child2, OpClasses)
    found3 = find_closest_children(node.child3, OpClasses)
    return found1.union(found2).union(found3)
  elif node.num_children == 2:
    lfound = find_closest_children(node.lchild, OpClasses)
    rfound = find_closest_children(node.rchild, OpClasses)
    return lfound.union(rfound)
  elif node.num_children == 1:
    return find_closest_children(node.child, OpClasses)
  return OrderedSet()


"""
returns whether ast has any learnable parameters
"""
@extclass(Node)
def has_parameters(self):
  learnables = [Conv1x1, Conv1D, Conv2D]
  preorder = self.preorder()
  for n in preorder:
    if type(n) in learnables:
      return True
  return False


# ops to choose from for insertion
linear_insert_ops = OrderedSet((Conv1x1, Conv1D, Conv2D))
nonlinear_insert_ops = OrderedSet((Relu,)) # only allow Softmax to be used with Mul insertion 
special_insert_ops = OrderedSet((Mul, Add, Sub, Stack, InterleavedSum, GroupedSum))

downsample_ops = OrderedSet((LearnedDownsample, Pack))
upsample_ops = OrderedSet((LearnedUpsample, BilinearUpsample, Unpack))

all_insert_ops = linear_insert_ops | nonlinear_insert_ops | special_insert_ops

linear_ops = OrderedSet((Conv1x1, Conv1D, Conv2D))

ops_with_changeable_channels = OrderedSet((LearnedDownsample, LearnedUpsample)) | linear_ops

special_linear_ops = OrderedSet((LearnedUpsample, LearnedDownsample))
nonlinear_ops = OrderedSet((Softmax, Relu)) 

border_ops = OrderedSet((RGBExtractor, XRGBExtractor, XFlatRGBExtractor, RGB8ChanExtractor, FlatRGB8ChanExtractor, \
                        GreenExtractor, XGreenExtractor, XFlatGreenExtractor, GreenRBExtractor, XGreenRBExtractor, SGreenExtractor, Flat2Quad, Input))

nas_ops = OrderedSet((Conv1x1, Conv2D, Relu, Softmax, Add, Stack))

all_ops = all_insert_ops | downsample_ops | upsample_ops | nonlinear_ops | border_ops
