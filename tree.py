"""
Functions for managing AST trees
"""
from abc import ABC, abstractmethod
import hashlib
import numpy as np


def make_tuple(x):
  try:
    iter(x)
  except TypeError:
    x = (x,)
  return x

def hash_combine(h, otherh):
  # from boost
  h = np.uint64(h)
  otherh = np.uint64(otherh)
  h ^= ( otherh + np.uint64(0x9e3779b9) + (h << np.uint64(6)) + (h >> np.uint64(2)) )
  return h

def myhash(value):
  if not type(value) is str:
    value = str(value).encode('utf-8')
  value = value.encode('utf-8')
  return int(hashlib.sha256(value).hexdigest()[:16], 16)-2**63


class Node:
  def __init__(self, name, num_children, parent=None):
    assert type(self) is not Node, "Do not try to instantiate abstract expressions"
    self.is_root = False
    self.parent = parent
    self.name = name
    self.num_children = num_children
    
  def __hash__(self):
    name_hash = myhash(self.__class__.__name__)

    h = hash_combine(name_hash, self.out_c)
    if type(self.in_c) is tuple:
      for c_idx in range(len(self.in_c)):
        h = hash_combine(h, self.in_c[c_idx])
    else: 
      h = hash_combine(h, self.in_c)
      
    if hasattr(self, "kwidth"):
      h = hash_combine(h, self.kwidth)
    if hasattr(self, "groups"):
      h = hash_combine(h, self.groups)
    if hasattr(self, "green_model_id"):
      h = hash_combine(h, self.green_model_id)

    if self.num_children == 3:
      h = hash_combine(h, hash(self.child1))
      h = hash_combine(h, hash(self.child2))
      h = hash_combine(h, hash(self.child3))
    elif self.num_children == 2:
      h = hash_combine(h, hash(self.lchild))
      h = hash_combine(h, hash(self.rchild))
    elif self.num_children == 1:
      h = hash_combine(h, hash(self.child))

    return int(h)

  def id_string(self):
    if self.num_children == 0:
      id_str = f"{self.name}-"
    else:
      id_str = f"{self.__class__.__name__}-"

    id_str += f"{self.out_c}-"

    if type(self.in_c) is tuple:
      if len(self.in_c) == 2:
        id_str += f"{self.in_c[0]};{self.in_c[1]}-"
      else:
        id_str += f"{self.in_c[0]};{self.in_c[1]};{self.in_c[2]}-"
    else:
      id_str += f"{self.in_c}-"
    if hasattr(self, "kwidth"):
      id_str += f"k{self.kwidth}-"
    if hasattr(self, "groups"):
      id_str += f"g{self.groups}-"
    if hasattr(self, "green_model_id"):
      id_str += f"green_model{self.green_model_id}-"
    if hasattr(self, "node"):
      id_str += f"{self.node.id_string()}-"
    if self.num_children == 3:
      id_str += f"{self.child1.id_string()}-"
      id_str += f"{self.child2.id_string()}-"
      id_str += f"{self.child3.id_string()}-"
    elif self.num_children == 2:
      id_str += f"{self.lchild.id_string()}-"
      id_str += f"{self.rchild.id_string()}-"
    elif self.num_children == 1:
      id_str += f"{self.child.id_string()}-"
    return id_str

    
  @abstractmethod
  def compute_input_output_channels(self):
    pass

  def dump(self, indent="", printstr="", nodeid=None):
    if nodeid is None:
      nodeid = 0
    tab = "   "
    if not hasattr(self, 'in_c'):
      self.compute_input_output_channels()
    printstr += "\n {} {} {} {}".format(indent, self.name, self.in_c, self.out_c)
    if hasattr(self, "groups"):
      printstr += f" g{self.groups}"
    if hasattr(self, "green_model_id"):
      printstr += f" green_model{self.green_model_id}"
    printstr += f"  [ID: {nodeid}] {id(self)}"

    nodeid += 1
    if self.num_children == 3:
      printstr = self.child1.dump(indent+tab, printstr, nodeid)

      child1size = self.child1.compute_size(set(), count_input_exprs=True)
      nodeid += child1size
      printstr = self.child2.dump(indent+tab, printstr, nodeid)

      child2size = self.child2.compute_size(set(), count_input_exprs=True)
      nodeid += child2size
      printstr = self.child3.dump(indent+tab, printstr, nodeid)

    elif self.num_children == 2:
      printstr = self.lchild.dump(indent+tab, printstr, nodeid)

      lchildsize = self.lchild.compute_size(set(), count_input_exprs=True)
      nodeid += lchildsize
      printstr = self.rchild.dump(indent+tab, printstr, nodeid)

    elif self.num_children == 1:
      printstr = self.child.dump(indent+tab, printstr, nodeid)

    elif hasattr(self, 'node') and not hasattr(self, 'green_model_id'):
      printstr = self.node.dump(indent+tab, printstr, nodeid)
      
    return printstr

  """
  if count_input_exprs=True count the size of the subtree
  used as Input to another tree
  """
  def compute_size(self, seen_inputs, count_all_inputs=False, count_input_exprs=False):
    preorder_nodes = self.preorder()
    if self.num_children == 3:
      children = [self.child1, self.child2, self.child3]
    elif self.num_children == 2:
      children = [self.lchild, self.rchild]
    elif self.num_children == 1:
      children = [self.child]
    else: 
      children = []
    for c in children:
      c.compute_size(seen_inputs, count_all_inputs, count_input_exprs)

    self.size = len(preorder_nodes)
    return self.size
  
  def preorder(self, nodes=None, seen=None):
    if seen is None:
      seen = set()
    if nodes is None:
      nodes = []

    if not id(self) in seen:
      nodes += [self]
      seen.add(id(self))

    if self.num_children == 3:
      self.child1.preorder(nodes, seen)
      self.child2.preorder(nodes, seen)
      self.child3.preorder(nodes, seen)
    elif self.num_children == 2:
      self.lchild.preorder(nodes, seen)
      self.rchild.preorder(nodes, seen)
    elif self.num_children == 1:
      self.child.preorder(nodes, seen)
    return nodes

  def get_preorder_id(self, node):
    preorder_nodes = self.preorder()
    for i,n in enumerate(preorder_nodes):
      if id(node) == id(n):
        return i
        
  def get_inputs(self, nodes=None):
    if nodes is None:
      nodes = set()
    if self.num_children == 0:
      if not self.name in nodes:
        nodes.add(self.name)
    elif self.num_children == 1:
      self.child.get_inputs(nodes)
    elif self.num_children == 2:
      self.lchild.get_inputs(nodes)
      self.rchild.get_inputs(nodes)
    elif self.num_children == 3:
      self.child1.get_inputs(nodes)
      self.child2.get_inputs(nodes)
      self.child3.get_inputs(nodes)
    else:
      assert False, "Invalid number of children"
    return nodes

  def add_dependee(self, node):
    if not hasattr(self, 'dependees'):
      self.dependees = set()
    self.dependees.add(node)

  def add_dependees(self):
    input_nodes = self.get_inputs()
    for n in input_nodes:
      if hasattr(n, 'node'):
        n.node.add_dependee(n)



  """
  assigns parents to all nodes in tree
  assumes parent field is None when called
  should only be called when a tree is first constructed
  """
  def assign_parents(self):
    if self.num_children == 3:
      for child in [self.child1, self.child2, self.child3]:
        if child.parent: # child has multiple parents
          child.parent = make_tuple(child.parent) + (self,) 
        else:
          child.parent = self
          child.assign_parents()
    elif self.num_children == 2:
      for child in [self.lchild, self.rchild]:
        if child.parent: # child has multiple parents
          child.parent = make_tuple(child.parent) + (self,) 
        else:
          child.parent = self
          child.assign_parents()
    elif self.num_children == 1:
      if self.child.parent:
        self.child.parent = make_tuple(self.child.parent) + (self,)
      else:
        self.child.parent = self
        self.child.assign_parents()


  def is_same_as_wrapper(self, other):
    return self.is_same_as(other, self, other)

  """
  returns whether or not two ASTs are the same
  """
  def is_same_as(self, other, root, other_root):
    if self.num_children != other.num_children or type(self) != type(other):
      return False

    if type(self.parent) is tuple:
      if not type(other.parent) is tuple:
        return False
      if not len(self.parent) == len(other.parent):
        return False
      parent_ids = set([root.get_preorder_id(p) for p in self.parent])
      other_parent_ids = set([other_root.get_preorder_id(p) for p in other.parent])

      if not parent_ids == other_parent_ids:
        return False

    if self.num_children == 0: # input nodes
      if self.out_c != other.out_c or self.in_c != other.in_c:
        return False
      if self.name != other.name:
        return False
      if hasattr(self, "green_model_id"):
        return self.green_model_id == other.green_model_id
      if hasattr(self, "node"):
        return (self.node).is_same_as_wrapper(other.node)
      return True
    elif self.num_children == 1: 
      if self.out_c != other.out_c or self.in_c != other.in_c:
        return False
      if hasattr(self, "groups"):
        if self.groups != other.groups:
          return False
      return self.child.is_same_as(other.child, root, other_root)
    elif self.num_children == 2:
      if self.out_c != other.out_c:
        return False      
      flipped_same = self.rchild.is_same_as(other.lchild, root, other_root) \
                and self.lchild.is_same_as(other.rchild, root, other_root)
      same = self.lchild.is_same_as(other.lchild, root, other_root) \
          and self.rchild.is_same_as(other.rchild, root, other_root) 
      return same or flipped_same
    elif self.num_children == 3:
      if self.out_c != other.out_c:
        return False
      # nodes with three children are not symmetric, order must be obeyed
      same = self.child1.is_same_as(other.child1, root, other_root) 
      same = same and self.child2.is_same_as(other.child2, root, other_root)
      same = same and self.child3.is_same_as(other.child3, root, other_root)
      return same 


  """
  returns whether or not two ASTs are the same - IGNORING channel count
  """
  def is_same_mod_channels(self, other):
    if self.num_children != other.num_children or type(self) != type(other):
      return False

    if self.num_children == 0:
      if self.out_c != other.out_c or self.in_c != other.in_c:
        return False
      if self.name != other.name:
        return False
      if hasattr(self.green_model_id):
        return self.green_model_id == other.green_model_id
      return True
    elif self.num_children == 1:
      return self.child.is_same_mod_channels(other.child)
    elif self.num_children == 2:
      return (self.lchild.is_same_mod_channels(other.lchild) and self.rchild.is_same_mod_channels(other.rchild))\
          or (self.rchild.is_same_mod_channels(other.lchild) and self.lchild.is_same_mod_channels(other.rchild))
    elif self.num_children == 3:
      same = self.child1.is_same_mod_channels(other.child1)
      same = same and self.child2.is_same_mod_channels(other.child2)
      same = same and self.child3.is_same_mod_channels(other.child3)
      return same

  """
  returns a list of the input and output channels of a tree in preorder
  """
  def get_input_output_channels(self):
    in_out_channels = []
    for n in self.preorder():
      in_out_channels.append((n.in_c, n.out_c))
    return in_out_channels

  """
  resets the input and output channels of a tree given a list of input output channels in preorder
  """
  def set_input_output_channels(self, in_out_channels):
    for i,n in enumerate(self.preorder()):
      n.in_c = in_out_channels[i][0]
      n.out_c = in_out_channels[i][1]

  def __eq__(self, other):
    return self.is_same_as_wrapper(other)

  def __ne__(self, other):
    return not self.is_same_as_wrapper(other)

"""
detects if there is loop in tree
"""
def has_loop(tree, seen=None):
  # loop if and node has a descendant that is also a parent
  if seen is None:
    seen = set()
  if id(tree) in seen:
    return True
  seen.add(id(tree))
  if tree.num_children == 3:
    return any([has_loop(c) for c in [tree.child1, tree.child2, tree.child3]]) 
  elif tree.num_children == 2:
    return has_loop(tree.lchild) or has_loop(tree.rchild)
  elif tree.num_children == 1:
    return has_loop(tree.child)
  return False


def print_parents(tree):
  parents = make_tuple(tree.parent)
  parent_ids = ""
  for p in parents:
    parent_ids += f"{id(p)} "

  print(f"{tree} {id(tree)} parents {parent_ids}")
  if tree.num_children == 3:
    for child in [tree.child1, tree.child2, tree.child3]:
      print_parents(child)
  elif tree.num_children == 2:
    for child in [tree.lchild, tree.rchild]:
      print_parents(child)
  elif tree.num_children == 1:
    print_parents(tree.child)


