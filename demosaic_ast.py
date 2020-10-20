"""
TODO:
  ADD GREEN AND CHROMA EXTRACTORS
"""
from abc import ABC, abstractmethod
from tree import Node, has_loop, hash_combine
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
      assert(out_c == node.out_c), "output channels of node to input doesn't match given out_c"
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
    self.out_c = None
    self.in_c = None

class Sub(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "Sub"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.out_c = None
    self.in_c = None

class Mul(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "Mul"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.out_c = None
    self.in_c = None

class LogSub(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "LogSub"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.out_c = None
    self.in_c = None

class AddExp(BinopIII, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "AddExp"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.out_c = None
    self.in_c = None

class Stack(BinopIJK, Special, Node):
  def __init__(self, lchild, rchild, name=None):
    if name is None:
      name = "Stack"
    Node.__init__(self, name, 2)
    self.lchild = lchild
    self.rchild = rchild
    self.out_c = None
    self.in_c = None
    
class ChromaExtractor(TernaryHcIcJcKc, Special, Node):
  def __init__(self, child1, child2, child3, name=None):
    if name is None:
      name = "ChromaExtractor"
    Node.__init__(self, name, 3)
    self.child1 = child1
    self.child2 = child2
    self.child3 = child3
    self.in_c = (3, 1, 1) # CH, CV, CQ, Bayer, Green
    self.out_c = 3
  def Hc(self):
    return 3 # ch chv, cq
  def Ic(self):
    return 1 # bayer
  def Jc(self):
    return 1 # green
  def Kc(self):
    return 3

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
    self.out_c = None
    self.in_c = None

class Relu(UnopII, NonLinear, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "Relu"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = None
    self.in_c = None

class Log(UnopII, NonLinear, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "Log"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = None
    self.in_c = None

class Exp(UnopII, NonLinear, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "Exp"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = None
    self.in_c = None

class Downsample(UnopII, Special, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "Downsample"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = None
    self.in_c = None

class Upsample(UnopII, Special, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "Upsample"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = None
    self.in_c = None

class FastUpsample(UnopII, Special, Node):
  def __init__(self, child, name=None):
    print("USING FAST UPSAMPLE")
    if name is None:
      name = "FastUpsample"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = None
    self.in_c = None

class Conv1x1(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int, name=None):
    if name is None:
      name = "Conv1x1"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = out_c
    self.in_c = None

class Conv1D(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int, name=None, kwidth=5):
    if name is None:
      name = "Conv1D"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = out_c
    self.kwidth = kwidth
    self.in_c = None

class Conv2D(UnopIJ, Linear, Node):
  def __init__(self, child, out_c: int, name=None, kwidth=5):
    if name is None:
      name = "Conv2D"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = out_c
    self.kwidth = kwidth
    self.in_c = None

class SumR(UnopI1, Special, Node):
  def __init__(self, child, name=None):
    if name is None:
      name = "SumR"
    Node.__init__(self, name, 1)
    self.child = child
    self.out_c = 1
    self.in_c = None


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
  self.child1.compute_input_output_channels()
  self.child2.compute_input_output_channels()
  self.child3.compute_input_output_channels()
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
    if type(n.parent) is tuple:
      for node_parent in n.parent:
        for j in range(0, i):
          if preorder[j] is node_parent:
            parents += [j]
    else:
      for j in range(0, i):
        if preorder[j] is n.parent:
          parents += [j]
    node_info["parent"] = parents

    if n.num_children == 3:
      child1_id = None
      child2_id = None
      child3_id = None
      for j in range(i, len(preorder)):
        if preorder[j] is n.child1:
          child1_id = j
        elif preorder[j] is n.child2:
          child2_id = j
        elif preorder[j] is n.child3:
          child3_id = j
        if not any([c is None for c in [child1_id, child2_id, child3_id]]):
          break
      node_info["children"] = [child1_id, child2_id, child3_id]

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
    

def build_tree_from_data(node_id, preorder_nodes, shared_children=None):
  if shared_children is None:
    shared_children = {}
  node_info = preorder_nodes[node_id]
  node_type = node_info["type"]
  node_class = str_to_class(node_type)
  node_name = node_info["name"]
  if "children" in node_info:
    children_ids = node_info["children"]
    if len(children_ids) == 3:
      child1_node = build_tree_from_data(children_ids[0], preorder_nodes, shared_children)
      child2_node = build_tree_from_data(children_ids[1], preorder_nodes, shared_children)
      child3_node = build_tree_from_data(children_ids[2], preorder_nodes, shared_children)
      new_node = node_class(child1_node, child2_node, child3_node, name=node_name)
    elif len(children_ids) == 2:
      lchild_node = build_tree_from_data(children_ids[0], preorder_nodes, shared_children)
      rchild_info = preorder_nodes[children_ids[1]]
      if len(rchild_info["parent"]) == 2:
        if children_ids[1] in shared_children:
          rchild_node = shared_children[children_ids[1]]
        else:
          rchild_node = build_tree_from_data(children_ids[1], preorder_nodes, shared_children)
          shared_children[children_ids[1]] = rchild_node
      else:
        rchild_node = build_tree_from_data(children_ids[1], preorder_nodes, shared_children)

      new_node = node_class(lchild_node, rchild_node, name=node_name)
    else:
      child_node = build_tree_from_data(children_ids[0], preorder_nodes, shared_children)
      if issubclass(node_class, UnopIJ):
        if "kwidth" in node_info:
          new_node = node_class(child_node, node_info["out_c"], name=node_name, kwidth=node_info["kwidth"])
        else:
          new_node = node_class(child_node, node_info["out_c"], name=node_name)
      else:
        new_node = node_class(child_node, name=node_name)
  else: # is input node
    new_node = node_class(node_info["in_c"], name=node_name)
  
  return new_node


def link_partners_in_reconstructed_tree(tree, preorder_node_info):
  preorder = tree.preorder()
  for i, node_info in enumerate(preorder_node_info):
    if "partner_set" in node_info:
      partner_ids = node_info["partner_set"] # list of preorder ids of partners
      partner_set = set([ (preorder[pid], id(preorder[pid])) for pid in partner_ids])
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
"""
if hasattr(root, "partner")
  if id(partner) in dic
    dic[id(partner)]["new_partner2"] = new_root
    new_root.partner = dic[id(partner)]["new_partner1"]
  else:
    dic[id(root)] = {"new_partner1": new_root}
    # call copy on children
    new_root.partner = dic[id(root)]["new_partner2"]

partner adds self ot dictionary     
"""
def copy_subtree_helper(root, shared_children=None):
  if shared_children is None:
    shared_children = {}
  new_root = copy.copy(root)
  if new_root.num_children == 3:
    c1copy = copy_subtree_helper(root.child1, shared_children)
    c2copy = copy_subtree_helper(root.child2, shared_children)
    c3copy = copy_subtree_helper(root.child3, shared_children)
    
    c1copy.parent = new_root
    c2copy.parent = new_root
    c3copy.parent = new_root

    new_root.child1 = c1copy
    new_root.child2 = c2copy
    new_root.child3 = c3copy

  elif new_root.num_children == 2:
    lcopy = copy_subtree_helper(root.lchild, shared_children)
    if "LogSub" in root.name or "AddExp" in root.name:    
      if root.rchild in shared_children:
        rcopy = shared_children[root.rchild]
        rcopy.parent = (rcopy.parent, new_root)
      else:
        rcopy = copy_subtree_helper(root.rchild, shared_children)
        shared_children[root.rchild] = rcopy
        rcopy.parent = new_root
    else:
      rcopy = copy_subtree_helper(root.rchild, shared_children)
      rcopy.parent = new_root

    new_root.lchild = lcopy
    new_root.rchild = rcopy
    lcopy.parent = new_root
    
  elif new_root.num_children == 1:
    child_copy = copy_subtree_helper(root.child, shared_children)
    new_root.child = child_copy
    child_copy.parent = new_root

  new_root.parent = None
  return new_root

def redirect_to_new_partners(old_tree, new_tree):
  old_tree_preorder = old_tree.preorder()
  new_tree_preorder = new_tree.preorder()
  old_tree_ids = [id(n) for n in old_tree_preorder]

  done = set()
  for new_node in new_tree_preorder:
    if id(new_node) in done:
      continue
    done.add(id(new_node))
    if hasattr(new_node, "partner_set"):
      found = False
      new_node_partner_set = set()
      for p in new_node.partner_set:
        if not p[1] in old_tree_ids: # partner of new node is not within the copied subtree -it's in the shared parent tree
          pass
          #print(f"{new_node} partner not in old tree ids")

          # new_node_partner = p

          # # remove all connections between partner node and old nodes 
          # partners_of_new_node_partner = set()
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

"""
Returns refernces to the closest parents belonging to the 
given opclasses.
"""
def find_closest_parents(node, OpClasses):
  if any([issubclass(type(node), oc) for oc in OpClasses]):
      return set( [ (node, id(node)) ] )
  if node.parent is None:
    return set()
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
    return set( [(node, id(node))] )
  if node.num_children == 3:
    found1 = find_closest_children(node.child1, OpClasses)
    found2 = find_closest_children(node.child2, OpClasses)
    found3 = find_closest_children(node.child3, OpClasses)
    return found1.union(found2).union(found3)
  if node.num_children == 2:
    lfound = find_closest_children(node.lchild, OpClasses)
    rfound = find_closest_children(node.rchild, OpClasses)
    return lfound.union(rfound)
  if node.num_children == 1:
    return find_closest_children(node.child, OpClasses)
  return set()


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


# if __name__ == "__main__":
#   from model_lib import multires_green_model
#   import argparse
#   import random
#   import numpy as np
#   from mutate import Mutator

#   parser = argparse.ArgumentParser("Demosaic")
#   parser.add_argument('--default_channels', type=int, default=16, help='num of output channels for conv layers')
#   parser.add_argument('--max_nodes', type=int, default=33, help='max number of nodes in a tree')
#   parser.add_argument('--min_subtree_size', type=int, default=2, help='minimum size of subtree in insertion')
#   parser.add_argument('--max_subtree_size', type=int, default=11, help='maximum size of subtree in insertion')
#   parser.add_argument('--structural_sim_reject', type=float, default=0.66, help='rejection probability threshold for structurally similar trees')
#   parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
#   parser.add_argument('--model_database_dir', type=str, default='model_database', help='path to save model statistics')
#   parser.add_argument('--database_save_freq', type=int, default=5, help='model database save frequency')
#   parser.add_argument('--save', type=str, default='SEARCH_MODELS', help='experiment name')
#   parser.add_argument('--seed', type=int, default=2, help='random seed')
#   parser.add_argument('--generations', type=int, default=20, help='model search generations')
#   parser.add_argument('--seed_model_file', type=str, help='')
#   parser.add_argument('--cost_tiers', type=str, help='list of tuples of cost tier ranges')
#   parser.add_argument('--tier_size', type=int, default=20, help='how many models to keep per tier')
#   parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
#   parser.add_argument('--mutation_failure_threshold', type=int, default=500, help='max number of tries to mutate a tree')
#   parser.add_argument('--delete_failure_threshold', type=int, default=25, help='max number of tries to find a node to delete')
#   args = parser.parse_args()
#   random.seed(args.seed)
#   np.random.seed(args.seed)

#   mutator = Mutator(args, None)
#   model = multires_green_model()
#   copy_model = copy_subtree(model)
#   preorder = copy_model.preorder()
#   print(copy_model.dump())
#   for i, n in enumerate(preorder):
#     print(f"node {i} {n.__class__.__name__}")
  
#   model_inputs = set(("Input(Bayer)",))
#   mutated_tree = mutator.insert_mutation(copy_model, model_inputs, insert_above_node_id=8, insert_op=list(special_ops)[0])
#   print(mutated_tree.dump())
