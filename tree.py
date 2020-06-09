"""
Functions for managing AST trees
"""
from abc import ABC, abstractmethod

class Node:
  def __init__(self, name, num_children, parent=None):
    assert type(self) is not Node, "Do not try to instantiate abstract expressions"
    self.parent = parent
    self.name = name
    self.num_children = num_children

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
  def compute_size(self, seen_inputs, count_input_exprs=False):
    self.size = 0
    if self.num_children == 2:
      lsize = self.lchild.compute_size(seen_inputs, count_input_exprs)
      rsize = self.rchild.compute_size(seen_inputs, count_input_exprs)
      self.size += lsize + rsize + 1
    elif self.num_children == 1:
      self.size += self.child.compute_size(seen_inputs, count_input_exprs) + 1
    elif self.num_children == 0: # is an Input, could be raw input or reuse of another subtree's output
      if hasattr(self, 'node'):
        if count_input_exprs:
          self.size = self.node.compute_size(seen_inputs, count_input_exprs)
        else:
          self.size = 0 # node copies output from another node, don't count it's compute
      elif not self in seen_inputs:
        self.size = 1
        seen_inputs.add(self)
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

  def get_inputs(self, nodes=None):
    if nodes is None:
      nodes = set()
    if self.num_children == 0:
      if not self in nodes:
        nodes.add(self)
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
  """
  def assign_parents(self):
    if self.num_children == 2:
      self.lchild.parent = self
      self.rchild.parent = self
      self.lchild.assign_parents()
      self.rchild.assign_parents()
    elif self.num_children == 1:
      self.child.parent = self
      self.child.assign_parents()


"""
Finds the closest ancestor of the given node that
belongs to a specified op class. If none are found, returns None
Assumes that assign_parents() has already been called on the 
full tree.
node: the node 
OpClass: the class 

returns all the nodes between the binop ancestor and the node 
including the binop ancestor and the node
"""
def find_closest_subclass_ancestor(node, OpClass):
  nodes = [node]
  while (not node.parent is None):
    node = node.parent
    nodes = [node] + nodes
    if issubclass(type(node), OpClass):
      break
  return nodes


