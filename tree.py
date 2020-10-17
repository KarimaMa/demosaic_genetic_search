"""
Functions for managing AST trees
"""
from abc import ABC, abstractmethod
import hashlib
import numpy as np


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
      h = hash_combine(h, self.in_c[0])
      h = hash_combine(h, self.in_c[1])
    else: 
      h = hash_combine(h, self.in_c)

    if self.num_children == 2:
      h = hash_combine(h, hash(self.lchild))
      h = hash_combine(h, hash(self.rchild))
    elif self.num_children == 1:
      h = hash_combine(h, hash(self.child))
    return int(h)

  def id_string(self):
    id_str = f"{self.__class__.__name__}-"
    id_str += f"{self.out_c}-"
    if type(self.in_c) is tuple:
      id_str += f"{self.in_c[0]};{self.in_c[1]}-"
    else:
      id_str += f"{self.in_c}-"
    if self.num_children == 2:
      id_str += f"{self.lchild.id_string()}-"
      id_str += f"{self.rchild.id_string()}-"
    elif self.num_children == 1:
      id_str += f"{self.child.id_string()}-"
    return id_str

    
  @abstractmethod
  def compute_input_output_channels(self):
    pass

  def dump(self, indent="", printstr=""):
    tab = "   "
    if not hasattr(self, 'in_c'):
      self.compute_input_output_channels()
    printstr += "\n {} {} {} {}".format(indent, self.name, self.in_c, self.out_c)
    if self.num_children == 2:
      printstr = self.lchild.dump(indent+tab, printstr)
      printstr = self.rchild.dump(indent+tab, printstr)
    elif self.num_children == 1:
      printstr = self.child.dump(indent+tab, printstr)
    return printstr

  """
  if count_input_exprs=True count the size of the subtree
  used as Input to another tree
  """
  def compute_size(self, seen_inputs, count_all_inputs=False, count_input_exprs=False):
    self.size = 0
    if self.num_children == 2:
      lsize = self.lchild.compute_size(seen_inputs, count_all_inputs, count_input_exprs)
      rsize = self.rchild.compute_size(seen_inputs, count_all_inputs, count_input_exprs)
      self.size += lsize + rsize + 1
    elif self.num_children == 1:
      self.size += self.child.compute_size(seen_inputs, count_all_inputs, count_input_exprs) + 1
    elif self.num_children == 0: # is an Input, could be raw input or reuse of another subtree's output
      if hasattr(self, 'node'):
        if count_input_exprs:
          self.size = self.node.compute_size(seen_inputs, count_all_inputs, count_input_exprs)
        else:
          self.size = 0 # node copies output from another node, don't count it's compute
      elif (not self.name in seen_inputs) or count_all_inputs:
        self.size = 1
        seen_inputs.add(self.name)
      else:
        self.size = 0 # already counted this input node
    else:
      assert False, "Invalid number of children"
    return self.size
  
  def preorder(self, nodes=None):
    if nodes is None:
      nodes = []

    nodes += [self]

    if self.num_children == 2:
      self.lchild.preorder(nodes)
      self.rchild.preorder(nodes)
    elif self.num_children == 1:
      self.child.preorder(nodes)
    return nodes

  def get_preorder_id(self, node):
    preorder_nodes = self.preorder()
    for i,n in enumerate(preorder_nodes):
      if node is n:
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
    if self.num_children == 2:
      if self.lchild.parent:
        # child has multiple parents:
        self.lchild.parent = (self.lchild.parent, self)
      else:
        self.lchild.parent = self
      if self.rchild.parent:
        # child has multiple parents:
        self.rchild.parent = (self.rchild.parent, self)
      else:
        self.rchild.parent = self
      self.lchild.assign_parents()
      self.rchild.assign_parents()
    elif self.num_children == 1:
      self.child.parent = self
      self.child.assign_parents()

  """
  returns whether or not two ASTs are the same
  """
  def is_same_as(self, other):
  
    if self.num_children != other.num_children or type(self) != type(other):
      return False
    if self.num_children == 0:
      if self.out_c != other.out_c or self.in_c != other.in_c:
        return False
      return self.name == other.name
    elif self.num_children == 1:
      if self.out_c != other.out_c or self.in_c != other.in_c:
        return False
      return self.child.is_same_as(other.child)
    elif self.num_children == 2:
      if self.out_c != other.out_c:
        return False      
      #if self.in_c[0] == other.in_c[1] and self.in_c[1] == other.in_c[0]:
      flipped_same = self.rchild.is_same_as(other.lchild) and self.lchild.is_same_as(other.rchild)
      #if self.in_c[0] == other.in_c[0] and self.in_c[1] == other.in_c[1]:
      same = self.lchild.is_same_as(other.lchild) and self.rchild.is_same_as(other.rchild) 
      return same or flipped_same

  """
  returns whether or not two ASTs are the same - IGNORING channel count
  """
  def is_same_mod_channels(self, other):
    if self.num_children != other.num_children or type(self) != type(other):
      return False

    if self.num_children == 0:
      if self.out_c != other.out_c or self.in_c != other.in_c:
        return False
      return self.name == other.name
    elif self.num_children == 1:
      return self.child.is_same_mod_channels(other.child)
    elif self.num_children == 2:
      return (self.lchild.is_same_mod_channels(other.lchild) and self.rchild.is_same_mod_channels(other.rchild))\
          or (self.rchild.is_same_mod_channels(other.lchild) and self.lchild.is_same_mod_channels(other.rchild))

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
    return self.is_same_as(other)

  def __ne__(self, other):
    return not self.is_same_as(other)

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
  if tree.num_children == 2:
    return has_loop(tree.lchild) or has_loop(tree.rchild)
  if tree.num_children == 1:
    return has_loop(tree.child)
  return False



