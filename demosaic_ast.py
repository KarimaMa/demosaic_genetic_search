"""
TODO:
  ADD GREEN AND CHROMA EXTRACTORS
"""
from abc import ABC, abstractmethod
from tree import Node



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

class Unop(ABC):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class UnopII(Unop):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class UnopIJ(Unop):
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"
      
class UnopI1(Unop):
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


class Input(Const, Special, Node):
  def __init__(self, out_c, name="Input", node=None):
    if node:
      name = "Input({})".format(node.name)
      self.node = node
    Node.__init__(self, name, 0)
    self.in_c = out_c
    self.out_c = out_c 

  def compute_input_output_channels(self):
    return self.in_c, self.out_c


class Add(BinopIII, Special, Node):
  def __init__(self, lchild, rchild):
    Node.__init__(self, "Add", 2)
    self.lchild = lchild
    self.rchild = rchild

  def compute_input_output_channels(self):
    leftchild_in_c, leftchild_out_c = self.lchild.compute_input_output_channels()
    rightchild_in_c, rightchild_out_c = self.rchild.compute_input_output_channels()
    self.in_c = (leftchild_out_c, rightchild_out_c)
    self.out_c = max(leftchild_out_c, rightchild_out_c)
    return self.in_c, self.out_c


class Sub(BinopIII, Special, Node):
  def __init__(self, lchild, rchild):
    Node.__init__(self, "Sub", 2)
    self.lchild = lchild
    self.rchild = rchild

  def compute_input_output_channels(self):
    leftchild_in_c, leftchild_out_c = self.lchild.compute_input_output_channels()
    rightchild_in_c, rightchild_out_c = self.rchild.compute_input_output_channels()
    self.in_c = (leftchild_out_c, rightchild_out_c)
    self.out_c = max(leftchild_out_c, rightchild_out_c)
    return self.in_c, self.out_c

class Mul(BinopIII, Special, Node):
  def __init__(self, lchild, rchild):
    Node.__init__(self, "Mul", 2)
    self.lchild = lchild
    self.rchild = rchild

  def compute_input_output_channels(self):
    leftchild_in_c, leftchild_out_c = self.lchild.compute_input_output_channels()
    rightchild_in_c, rightchild_out_c = self.rchild.compute_input_output_channels()
    self.in_c = (leftchild_out_c, rightchild_out_c)
    self.out_c = max(leftchild_out_c, rightchild_out_c)
    return self.in_c, self.out_c


class LogSub(BinopIII, NonLinear, Node):
  def __init__(self, lchild, rchild):
    Node.__init__(self, "LogSub", 2)
    self.lchild = lchild
    self.rchild = rchild

  def compute_input_output_channels(self):
    leftchild_in_c, leftchild_out_c = self.lchild.compute_input_output_channels()
    rightchild_in_c, rightchild_out_c = self.rchild.compute_input_output_channels()
    self.in_c = (leftchild_out_c, rightchild_out_c)
    self.out_c = max(leftchild_out_c, rightchild_out_c)
    return self.in_c, self.out_c


class AddExp(BinopIII, NonLinear, Node):
  def __init__(self, lchild, rchild):
    Node.__init__(self, "AddExp", 2)
    self.lchild = lchild
    self.rchild = rchild

  def compute_input_output_channels(self):
    leftchild_in_c, leftchild_out_c = self.lchild.compute_input_output_channels()
    rightchild_in_c, rightchild_out_c = self.rchild.compute_input_output_channels()
    self.in_c = (leftchild_out_c, rightchild_out_c)
    self.out_c = max(leftchild_out_c, rightchild_out_c)
    return self.in_c, self.out_c


class Stack(BinopIJK, Special, Node):
  def __init__(self, lchild, rchild):
    Node.__init__(self, "Stack", 2)
    self.lchild = lchild
    self.rchild = rchild

  def compute_input_output_channels(self):
    _, lout_c = self.lchild.compute_input_output_channels()
    _, rout_c = self.rchild.compute_input_output_channels()
    self.in_c = (lout_c, rout_c)
    self.out_c = lout_c + rout_c
    return self.in_c, self.out_c


class ChromaExtractor(BinopIcJcKc, Special, Node):
  def __init__(self, lchild, rchild):
    Node.__init__(self, "ChromaExtractor", 2)
    self.lchild = lchild
    self.rchild = rchild
    self.in_c = (3, 1) # CH, CV, CQ and Bayer
    self.out_c = 2

  def compute_input_output_channels(self):
    self.lchild.compute_input_output_channels()
    self.rchild.compute_input_output_channels()
    return self.in_c, self.out_c

  def Ic(self):
    return 3
  def Jc(self):
    return 1
  def Kc(self):
    return 2


class GreenExtractor(BinopIcJcKc, Special, Node):
  def __init__(self, lchild, rchild):
    Node.__init__(self, "GreenExtractor", 2)
    self.lchild = lchild
    self.rchild = rchild
    self.in_c = (1, 1) # G and Bayer
    self.out_c = 1

  def compute_input_output_channels(self):
    self.lchild.compute_input_output_channels()
    self.rchild.compute_input_output_channels()
    return self.in_c, self.out_c

  def Ic(self):
    return 1
  def Jc(self):
    return 1
  def Kc(self):
    return 1


class Softmax(UnopII, NonLinear, Node):
  def __init__(self, child):
    Node.__init__(self, "Softmax", 1)
    self.child = child

  def compute_input_output_channels(self):
    _, lout_c = self.child.compute_input_output_channels()
    self.in_c = lout_c
    self.out_c = lout_c
    return self.in_c, self.out_c
    

class Relu(UnopII, NonLinear, Node):
  def __init__(self, child):
    Node.__init__(self, "Relu", 1)
    self.child = child

  def compute_input_output_channels(self):
    _, lout_c = self.child.compute_input_output_channels()
    self.in_c = lout_c
    self.out_c = lout_c
    return self.in_c, self.out_c


class Log(UnopII, NonLinear, Node):
  def __init__(self, child):
    Node.__init__(self, "Log", 1)
    self.child = child

  def compute_input_output_channels(self):
    _, lout_c = self.child.compute_input_output_channels()
    self.in_c = lout_c
    self.out_c = lout_c
    return self.in_c, self.out_c


class Exp(UnopII, NonLinear, Node):
  def __init__(self, child):
    Node.__init__(self, "Exp", 1)
    self.child = child

  def compute_input_output_channels(self):
    _, lout_c = self.child.compute_input_output_channels()
    self.in_c = lout_c
    self.out_c = lout_c
    return self.in_c, self.out_c


class Downsample(UnopII, Special, Node):
  def __init__(self, child):
    Node.__init__(self, "Downsample", 1)
    self.child = child

  def compute_input_output_channels(self):
    _, lout_c = self.child.compute_input_output_channels()
    self.in_c = lout_c
    self.out_c = lout_c
    return self.in_c, self.out_c


class Upsample(UnopII, Special, Node):
  def __init__(self, child):
    Node.__init__(self, "Upsample", 1)
    self.child = child

  def compute_input_output_channels(self):
    _, lout_c = self.child.compute_input_output_channels()
    self.in_c = lout_c
    self.out_c = lout_c
    return self.in_c, self.out_c


class Conv1x1(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int):
    Node.__init__(self, "Conv1x1", 1)
    self.child = child
    self.out_c = out_c

  def compute_input_output_channels(self):
    child_in_c, child_out_c = self.child.compute_input_output_channels()
    self.in_c = child_out_c
    return self.in_c, self.out_c


class Conv1D(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int):
    Node.__init__(self, "Conv1D", 1)
    self.child = child
    self.out_c = out_c

  def compute_input_output_channels(self):
    child_in_c, child_out_c = self.child.compute_input_output_channels()
    self.in_c = child_out_c
    return self.in_c, self.out_c


class Conv2D(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int):
    Node.__init__(self, "Conv2D", 1)
    self.child = child
    self.out_c = out_c

  def compute_input_output_channels(self):
    child_in_c, child_out_c = self.child.compute_input_output_channels()
    self.in_c = child_out_c
    return self.in_c, self.out_c


class SumR(UnopI1, Special, Node):
  def __init__(self, child):
    Node.__init__(self, "SumR", 1)
    self.child = child
    self.out_c = 1

  def compute_input_output_channels(self):
    _, child_out_c = self.child.compute_input_output_channels()
    self.in_c = child_out_c
    return self.in_c, self.out_c

linear_ops = set((Conv1x1, Conv1D, Conv2D))
nonlinear_ops = set((Softmax, Relu, Log)) #TODO ADD HANDLING FOR LOGSUB
special_ops = set((Mul, Add, Sub, Stack, Downsample, SumR))
sandwich_ops = set((Log, Downsample)) # ops that must be used with their counterparts (Exp, AddExp, Upsample)
sandwich_pairs = {
  Log: Exp,
  Downsample: Upsample
}

nl_and_sp = nonlinear_ops.union(special_ops)
l_and_sp = linear_ops.union(special_ops)
all_ops = nl_and_sp.union(l_and_sp)
