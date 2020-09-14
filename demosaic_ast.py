"""
TODO:
  ADD GREEN AND CHROMA EXTRACTORS
"""
from abc import ABC, abstractmethod
from tree import Node, hash_combine
import copy
import sys
import pickle


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
      name = node.name
      self.node = node
    Node.__init__(self, "Input({})".format(name), 0)
    self.in_c = out_c
    self.out_c = out_c 

class Add(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "Add"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild

class Sub(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "Sub"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild

class Mul(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "Mul"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild

class LogSub(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "LogSub"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild

class AddExp(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "AddExp"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild

class Stack(BinopIJK, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "Stack"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild

class ChromaExtractor(BinopIcJcKc, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "ChromaExtractor"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.in_c = (3, 1) # CH, CV, CQ and Bayer
    self.out_c = 2
  def Ic(self):
    return 3
  def Jc(self):
    return 1
  def Kc(self):
    return 2

class GreenExtractor(BinopIcJcKc, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "GreenExtractor"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.in_c = (1, 1) # G and Bayer
    self.out_c = 1
  def Ic(self):
    return 1
  def Jc(self):
    return 1
  def Kc(self):
    return 1

class Softmax(UnopII, NonLinear, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "Softmax"
    Node.__init__(self, name, 1)
    self.child = child

class Relu(UnopII, NonLinear, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "Relu"
    Node.__init__(self, name, 1)
    self.child = child

class Log(UnopII, NonLinear, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "Log"
    Node.__init__(self, name, 1)
    self.child = child

class Exp(UnopII, NonLinear, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "Exp"
    Node.__init__(self, name, 1)
    self.child = child

class Downsample(UnopII, Special, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "Downsample"
    Node.__init__(self, name, 1)
    self.child = child

class Upsample(UnopII, Special, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "Upsample"
    Node.__init__(self, name, 1)
    self.child = child

class Conv1x1(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int, name=None):
    if name is None:
      name = "Conv1x1"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = out_c

class Conv1D(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int, name=None, kwidth=5):
    if name is None:
      name = "Conv1D"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = out_c
    self.kwidth = kwidth

class Conv2D(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int, name=None, kwidth=5):
    if name is None:
      name = "Conv2D"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = out_c
    self.kwidth = kwidth

class SumR(UnopI1, Special, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "SumR"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = 1


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

@extclass(ChromaExtractor)
def compute_input_output_channels(self):
  self.lchild.compute_input_output_channels()
  self.rchild.compute_input_output_channels()
  return self.in_c, self.out_c

@extclass(GreenExtractor)
def compute_input_output_channels(self):
  self.lchild.compute_input_output_channels()
  self.rchild.compute_input_output_channels()
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

@extclass(Downsample)
def compute_input_output_channels(self):
  _, lout_c = self.child.compute_input_output_channels()
  self.in_c = lout_c
  self.out_c = lout_c
  return self.in_c, self.out_c

@extclass(Upsample)
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

@extclass(SumR)
def compute_input_output_channels(self):
  _, child_out_c = self.child.compute_input_output_channels()
  self.in_c = child_out_c
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
    if n.num_children == 2:
      lchild_id = None
      rchild_id = None
      for j in range(i, len(preorder)):
        if preorder[j] is n.lchild:
          lchild_id = j
        elif preorder[j] is n.rchild:
          rchild_id = j
        if not lchild_id is None and not rchild_id is None:
          break
      node_info["children"] = [lchild_id, rchild_id]

    elif n.num_children == 1:
      child_id = None
      for j in range(i, len(preorder)):
        if preorder[j] is n.child:
          child_id = j
          break
      node_info["children"] = [child_id]
  
    if hasattr(n, 'name'):
      node_info["name"] = n.name.lstrip("Input(").rstrip(")")
    if hasattr(n, 'kwidth'):
      node_info["kwidth"] = n.kwidth

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
    

def build_tree_from_data(node_id, preorder_nodes):
  node_info = preorder_nodes[node_id]
  node_type = node_info["type"]
  node_class = str_to_class(node_type)
  node_name = node_info["name"]
  if "children" in node_info:
    children_ids = node_info["children"]
    if len(children_ids) == 2:
      lchild_node = build_tree_from_data(children_ids[0], preorder_nodes)
      rchild_node = build_tree_from_data(children_ids[1], preorder_nodes)
      new_node = node_class(lchild_node, rchild_node, name=node_name)
    else:
      child_node = build_tree_from_data(children_ids[0], preorder_nodes)
      if issubclass(node_class, UnopIJ):
        if "kwidth" in node_info:
          new_node = node_class(child_node, node_info["kwidth"], name=node_name, kwidth=node_info["kwidth"])
        else:
          new_node = node_class(child_node, node_info["out_c"], name=node_name)
      else:
        new_node = node_class(child_node, name=node_name)
  else: # is input node
    new_node = node_class(node_info["in_c"], name=node_name)

  return new_node

def load_ast(filename):
  with open(filename, "rb") as f:
    tree_data = pickle.load(f)
  tree = build_tree_from_data(0, tree_data)
  tree.assign_parents()
  tree.compute_input_output_channels()
  return tree

def instance_to_classnamme(o):
  return o.__class__.__name__

def str_to_class(str):
  return getattr(sys.modules[__name__], str)


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

  op_list = [Conv1x1, Conv1D, Conv2D, Softmax, Relu, Mul, Add, Sub, AddExp, LogSub, Stack, Upsample, Downsample, SumR]
  ops_used = set([type(n) for n in nodes])
  op_coverage = 0
  for i,o in enumerate(op_list):
    if o in ops_used:
      op_coverage += 1 << i

  sh = hash_combine(h, op_coverage)
  sh = hash_combine(sh, len(nodes))
  return sh
  
"""
returns a deep copy of the subtree at root but does NOT
make a deep copy of ancestors. and deletes original references 
to the parent tree
"""
def copy_subtree(root):
  new_root = copy.copy(root)
  if new_root.num_children == 2:
    lcopy = copy.deepcopy(new_root.lchild)
    rcopy = copy.deepcopy(new_root.rchild)
    new_root.lchild = lcopy
    new_root.rchild = rcopy
    lcopy.parent = new_root
    rcopy.parent = new_root
  elif new_root.num_children == 1:
    child_copy = copy.deepcopy(new_root.child)
    new_root.child = child_copy
    child_copy.parent = new_root

  new_root.parent = None
  return new_root


"""
Finds the closest ancestor of the given node that inherits from any
class in the set OpClasses.
If we reach a node with multiple parents before we find the ancestor,
stop and return the tuple of parents istead.
NOTE: currently if a node has multiple parents it is always the 
case that the parents are Binops but this might not be true
in the future.
Assumes assign_parents() has already been called on the full tree.
Returns all the nodes between the found ancestor and the node 
including the ancestor and the node

"""
def find_closest_ancestor(node, OpClasses):
  nodes = [node]
  while (not node.parent is None):
    nodes = [node.parent] + nodes
    parent_type = type(node.parent)
    if parent_type is tuple or any([issubclass(parent_type, oc) for oc in OpClasses]):
      break
    node = node.parent
  return nodes




# ops to choose from for insertion
linear_insert_ops = set((Conv1x1, Conv1D, Conv2D))
nonlinear_insert_ops = set((Relu,)) # only allow Softmax to be used with Mul insertion 
special_insert_ops = set((Mul, Add, Sub, Stack, Downsample, SumR, LogSub))

nl_sp_insert_ops = nonlinear_insert_ops.union(special_insert_ops)
l_sp_insert_ops = linear_insert_ops.union(special_insert_ops)
all_insert_ops = nl_sp_insert_ops.union(l_sp_insert_ops)


linear_ops = set((Conv1x1, Conv1D, Conv2D))
nonlinear_ops = set((Softmax, Relu)) 
special_ops = set((Mul, Add, Sub, AddExp, LogSub, Stack, Upsample, Downsample, SumR))
sandwich_ops = set((LogSub, AddExp, Downsample, Upsample)) # ops that must be used with their counterparts (Exp, AddExp, Upsample)
sandwich_pairs = {
  LogSub: AddExp,
  Downsample: Upsample
}

border_ops = set((ChromaExtractor, GreenExtractor, Input))
nl_and_sp = nonlinear_ops.union(special_ops)
l_and_sp = linear_ops.union(special_ops)
all_ops = nl_and_sp.union(l_and_sp)


if __name__ == "__main__":
  x = str_to_class("Add")
  i1 = Input(1, "Bayer")
  i2 = Input(1, "Bayer")
  new = x(i1, i2)
  print(new)
  print(new.compute_input_output_channels())
  print(new.__class__.__name__)

