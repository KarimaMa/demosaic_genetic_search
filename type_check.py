from demosaic_ast import *
from enum import Enum 
import logging

logger = logging.getLogger("DebugLogger")

class Resolution(Enum):
  FULL = 1
  DOWNSAMPLED = 2
  INVALID = 3

class Layout(Enum):
  FLAT = 1
  QUAD = 2
  INVALID = 3

"""
attempts to shrink down channel counts of nodes 
as much as possible
"""
def shrink_channels(node, target_c, out_c=None):
  if isinstance(node, BinopIJK):
    if out_c is None:
      left_in_c = shrink_channels(node.lchild, target_c)
      right_in_c = shrink_channels(node.rchild, target_c)
      node.in_c = (left_in_c, right_in_c)
      node.out_c = left_in_c + right_in_c
    else:
      left_in_c = out_c // 2
      right_in_c = out_c - left_in_c
      shrink_channels(node.lchild, target_c, out_c=left_in_c)
      shrink_channels(node.rchild, target_c, out_c=right_in_c)
      node.in_c = (left_in_c, right_in_c)
      node.out_c = out_c
    return node.out_c
  elif isinstance(node, BinopIII):
    layout_t = layout_type(node)
    if layout_t == Layout.QUAD:
      broadcast_width = 4
    else:
      broadcast_width = 1

    if out_c is None:
      left_in_c = shrink_channels(node.lchild, target_c) 
      right_in_c = shrink_channels(node.rchild, target_c)

      assert left_in_c == right_in_c or (left_in_c == broadcast_width) or (right_in_c == broadcast_width), \
        f"Shrinking channels cannot make BinopIII left {left_in_c} match right {right_in_c} channels"
      node.out_c = left_in_c
      node.in_c = (left_in_c, right_in_c)
      return node.out_c
    else:
      try: # trying (out_c, out_c)
        shrink_channels(node.lchild, target_c, out_c=out_c)
        shrink_channels(node.rchild, target_c, out_c=out_c)
      except AssertionError:
        pass
      else:
        node.in_c = (out_c, out_c)
        node.out_c = out_c
        return node.out_c
      try: # trying (out_c, broadcast_c)
        shrink_channels(node.lchild, target_c, out_c=out_c)
        shrink_channels(node.rchild, target_c, out_c=broadcast_width)
      except AssertionError:
        pass
      else:
        node.in_c = (out_c, broadcast_width)
        node.out_c = out_c
        return node.out_c  
      try: # trying (broadcast_c, out_c)
        shrink_channels(node.lchild, target_c, out_c=broadcast_width)
        shrink_channels(node.rchild, target_c, out_c=out_c)
      except AssertionError:
        assert False, f"Failed to make BinopIII output channels agree with {out_c}"
      else:
        node.in_c = (broadcast_width, out_c)
        node.out_c = out_c
        return node.out_c  
  
  elif isinstance(node, BinopIcJcKc): # input and output channels are immutable
    if not out_c is None:
      assert out_c == node.Kc(), f"output channels of {node.__class__.__name__} cannot be set to {out_c}"
    shrink_channels(node.lchild, target_c, out_c=node.Ic())
    shrink_channels(node.rchild, target_c, out_c=node.Jc())
    return node.out_c
  elif isinstance(node, TernaryHcIcJcKc):
    if not out_c is None:
      assert out_c == node.Kc(), f"output channels of {node.__class__.__name__} cannot be set to {out_c}"
    shrink_channels(node.child1, target_c, out_c=node.Hc())
    shrink_channels(node.child2, target_c, out_c=node.Ic())
    shrink_channels(node.child3, target_c, out_c=node.Jc()) 
    return node.out_c  
  elif isinstance(node, UnopIcIc):
    if not out_c is None:
      assert out_c == node.Ic(), f"output channels of {node.__class__.__name__} cannot be set to {out_c}"
    shrink_channels(node.child, target_c, out_c=node.Ic())
    return node.out_c
  elif isinstance(node, UnopIJ):
    if out_c is None:
      node.out_c = min(node.out_c, target_c)
      node.in_c = shrink_channels(node.child, target_c)
    else:
      node.out_c = out_c
      node.in_c = shrink_channels(node.child, target_c)
    return node.out_c
  elif isinstance(node, UnopII):
    if out_c is None:
      node.in_c = shrink_channels(node.child, target_c)
      node.out_c = node.in_c
    else:
      shrink_channels(node.child, target_c, out_c=out_c)
      node.out_c = out_c
      node.in_c = out_c
    return node.out_c
  elif isinstance(node, UnopI1):
    if out_c is None:
      node.in_c = shrink_channels(node.child, target_c)
      node.out_c = 1
    else:
      assert out_c == 1, f"output channels of UnopI1 cannot be set to {out_c}"
      node.in_c = shrink_channels(node.child, target_c)
    return node.out_c
  elif isinstance(node, Const):
    if not out_c is None:
      assert node.out_c == out_c, f"Cannot set output channels of {node.name} to {out_c}"
    return node.out_c 
  else:
    assert False, "Unknown node type"


"""
returns the layout type of a subtree
"""
def layout_type(subtree):
  typ = None
  if isinstance(subtree, QuadExtractor):
    typ = Layout.FLAT
  elif isinstance(subtree, QuadInput):
    typ = Layout.QUAD
  elif isinstance(subtree, Input):
    typ = Layout.FLAT

  if subtree.num_children == 4:
    assert False, "Layout type for node with 4 children unimplemented"
  elif subtree.num_children == 3:
    typ1 = layout_type(subtree.child1)
    typ2 = layout_type(subtree.child2)
    typ3 = layout_type(subtree.child3)
    if typ1 != typ2 or typ1 != typ3:
      return Layout.INVALID
    child_typ = typ1
  elif subtree.num_children == 2:
    ltyp = layout_type(subtree.lchild)
    rtyp = layout_type(subtree.rchild)
    if ltyp != rtyp:
      return Layout.INVALID
    child_typ = ltyp
  elif subtree.num_children == 1:
    child_typ = layout_type(subtree.child)
  else: # input node
    child_typ = None
  
  if child_typ == Layout.INVALID:
    return Layout.INVALID

  if typ is None: # resolution obeys children 
    return child_typ
  else: # this node is either quad extractor or inut node, child cannot have same resolution
    if typ == child_typ:
      return Layout.INVALID
    return typ


"""
returns the spatial resolution of a subtree
"""
def spatial_resolution(subtree):
  res = None # resolution obeys children unless this node is an up or downsample
  if isinstance(subtree, Upsample):
    res = Resolution.FULL
  elif isinstance(subtree, Downsample):
    res = Resolution.DOWNSAMPLED

  if subtree.num_children == 4:
    res1 = spatial_resolution(subtree.child1)
    res2 = spatial_resolution(subtree.child2)
    res3 = spatial_resolution(subtree.child3)
    res4 = spatial_resolution(subtree.child4)

    if len(set([res1, res2, res3, res4])) != 1:
      return Resolution.INVALID
    child_res = res1

  elif subtree.num_children == 3:
    res1 = spatial_resolution(subtree.child1)
    res2 = spatial_resolution(subtree.child2)
    res3 = spatial_resolution(subtree.child3)
    if res1 != res2 or res1 != res3:
      return Resolution.INVALID
    child_res = res1

  elif subtree.num_children == 2:
    lres = spatial_resolution(subtree.lchild)
    rres = spatial_resolution(subtree.rchild)
    if lres != rres :
      return Resolution.INVALID
    child_res = lres

  elif subtree.num_children == 1:
    child_res = spatial_resolution(subtree.child)

  else:
    return Resolution.FULL # Input nodes have full resolution

  if child_res == Resolution.INVALID:
    return Resolution.INVALID

  if res is None: # resolution obeys children 
    return child_res
  else:
    if res == child_res: # this node up or downsamples, child must have opposite resolution
      return Resolution.INVALID
    return res
    
"""
checks that channels counts of AST agree across nodes
and returns output channels of the root node
"""
def check_channel_count(node):
  if isinstance(node, BinopIII):
    layout_t = layout_type(node)
    if layout_t == Layout.QUAD:
      broadcast_width = 4
    else:
      broadcast_width = 1

    lchild_c = check_channel_count(node.lchild)
    rchild_c = check_channel_count(node.rchild)
    assert(lchild_c == rchild_c or rchild_c == broadcast_width or lchild_c == broadcast_width) # allow broadcasting
    return max(rchild_c, lchild_c)
  elif isinstance(node, BinopIJK):
    lchild_c = check_channel_count(node.lchild)
    rchild_c = check_channel_count(node.rchild)
    out_c = lchild_c + rchild_c
    return out_c
  elif isinstance(node, BinopIcJcKc):
    lchild_c = check_channel_count(node.lchild)
    rchild_c = check_channel_count(node.rchild)
    assert(lchild_c == node.Ic() and rchild_c == node.Jc())
    return node.Kc()
  elif isinstance(node, TernaryHcIcJcKc):
    child1_c = check_channel_count(node.child1)
    child2_c = check_channel_count(node.child2)
    child3_c = check_channel_count(node.child3)
    assert(child1_c == node.Hc() and child2_c == node.Ic() and child3_c == node.Jc())
    return node.Kc()
  elif isinstance(node, UnopIcIc):
    child_c = check_channel_count(node.child)
    assert(child_c == node.Ic())
    return node.Ic()
  elif isinstance(node, UnopII):
    child_c = check_channel_count(node.child)
    return child_c
  elif isinstance(node, UnopIJ):
    check_channel_count(node.child)
    return node.out_c
  elif isinstance(node, UnopI1):
    check_channel_count(node.child)
    return 1
  elif isinstance(node, Const):
    return node.out_c

"""
returns whether given type exists between two nodes
"""
def find_type_between(top, bottom, T):
  if bottom is top:
    return False
  elif isinstance(bottom, T):
    return True
  else:
    return find_type_between(top, bottom.parent, T)
    
"""
returns list of earliest occurence(s) of type in tree
and the level of occurence
"""
def find_type(root, T, level, ignore_root=False):
  if isinstance(root, T) and not ignore_root:
    return [root], level
  elif isinstance(root, Const): # reached leaf node
    return [None], 1e10

  elif root.num_children == 4:
    found1, level1 = find_type(root.child1, T, level+1)
    found2, level2 = find_type(root.child2, T, level+1)
    found3, level3 = find_type(root.child3, T, level+1)
    found4, level4 = find_type(root.child4, T, level+1)

    found1 = list(filter(None, found1))
    found2 = list(filter(None, found2))
    found3 = list(filter(None, found3))
    found4 = list(filter(None, found4))

    found = [found1, found2, found3, found4]
    level = np.array([level1, level2, level3, level4])

    idx = np.argmin(level)
    return found[idx], level[idx]

  elif root.num_children == 3:
    found1, level1 = find_type(root.child1, T, level+1)
    found2, level2 = find_type(root.child2, T, level+1)
    found3, level3 = find_type(root.child3, T, level+1)
    found1 = list(filter(None, found1))
    found2 = list(filter(None, found2))
    found3 = list(filter(None, found3))

    found = [found1, found2, found3]
    level = np.array([level1, level2, level3])

    idx = np.argmin(level)
    return found[idx], level[idx]

  elif root.num_children == 2:
    lfound, leftl = find_type(root.lchild, T, level+1)
    rfound, rightl = find_type(root.rchild, T, level+1)
  
    lfound = list(filter(None, lfound))
    rfound = list(filter(None, rfound))

    if any(lfound) and any(rfound):
      if leftl < rightl:
        return lfound, leftl
      elif leftl > rightl:
        return rfound, rightl
      else:
        return lfound+rfound, leftl
    elif any(lfound):
      return lfound, leftl
    elif any(rfound):
      return rfound, rightl 
    else:
      return [None], 1e10
  else:
    found, l = find_type(root.child, T, level+1)
    return found, l


"""
attempts to fix output channels of given ast to match out_c
by finding topmost UnopIJ nodes and changing their output channels to out_c
"""
def fix_channel_count_downwards(root, out_c, fixed_shared_children=None):
  if fixed_shared_children is None:
    fixed_shared_children = {}

  if id(root) in fixed_shared_children: # can't change channels - check it matches required out_c
    fixed_out_c = fixed_shared_children[id(root)].out_c

    if fixed_out_c != out_c:
      return False
    return True
  else: # node is either not shared child or is being seen for the first time
    if isinstance(root, BinopIII):
      layout_t = layout_type(node)
      if layout_t == Layout.QUAD:
        broadcast_width = 4
      else:
        broadcast_width = 1

      lfixed = False
      rfixed = False

      lfixed = fix_channel_count_downwards(root.lchild, out_c, fixed_shared_children)
      if not lfixed: # try to make left channels = 1 and right channels = out_c 
        lfixed = fix_channel_count_downwards(root.lchild, broadcast_width, fixed_shared_children)
        if lfixed:
          rfixed = fix_channel_count_downwards(root.rchild, out_c, fixed_shared_children)
      else: # left channels = out_c
        rfixed = fix_channel_count_downwards(root.rchild, out_c, fixed_shared_children)
        if not rfixed:
          rfixed = fix_channel_count_downwards(root.rchild, broadcast_width, fixed_shared_children)
        
      if rfixed and type(root.parent) is tuple:
        fixed_shared_children[id(root)] = root
      
      return rfixed

    elif isinstance(root, BinopIJK):
      lchild_out_c = out_c // 2
      if out_c % 2 == 1:
        rchild_out_c = lchild_out_c + 1
      else:
        rchild_out_c = lchild_out_c
      if lchild_out_c == 0:
        return False
      # naively divide out_c evenly across inputs
      lfixed = fix_channel_count_downwards(root.lchild, lchild_out_c, fixed_shared_children)
      if lfixed:
        rfixed = fix_channel_count_downwards(root.rchild, rchild_out_c, fixed_shared_children)
        if rfixed and type(root.parent) is tuple:
          fixed_shared_children[id(root)] = root
        return rfixed
      else:
        return False
    elif isinstance(root, UnopIJ):
      root.out_c = out_c
      if type(root.parent) is tuple:
        fixed_shared_children[id(root)] = root
      fixed = fix_channel_count_downwards(root.child, root.in_c, fixed_shared_children)
      return fixed 
    elif isinstance(root, BinopIcJcKc) or isinstance(root, UnopI1) or isinstance(root, Const):
      if root.out_c != out_c:
        return False
      else:
        if type(root.parent) is tuple:
          fixed_shared_children[id(root)] = root
        return True
    elif isinstance(root, TernaryHcIcJcKc) or isinstance(root, UnopIcIc):
      if root.out_c != out_c:
        return False
      return True
    else: # is type UnopII
      fixed = fix_channel_count_downwards(root.child, out_c, fixed_shared_children)
      if fixed and type(root.parent) is tuple:
        fixed_shared_children[id(root)] = root
      return fixed

"""
attempts to fix input channels of parent tree to match output channels of subtree
by finding closest UnopIJ ancestor and changing its input channels to out_c
if a Binop is encountered, fix the channel counts downwards for the other child
if the requried input channels is not 1.
"""
def fix_channel_count_upwards_helper(subtree, parent, in_c):
  cur_node = subtree
  if parent is None:
    return True
  elif isinstance(parent, BinopIII):
    layout_t = layout_type(node)
    if layout_t == Layout.QUAD:
      broadcast_width = 4
    else:
      broadcast_width = 1

    if in_c == broadcast_width: # don't need to change other child, Binops can broadcast
      return True

    if cur_node is parent.lchild:
      child_fixed = fix_channel_count_downwards(parent.rchild, in_c)
      if not child_fixed: # allowed to broadcast so 1 is also a valid channel count
        child_fixed = fix_channel_count_downwards(parent.rchild, broadcast_width)
    else:
      child_fixed = fix_channel_count_downwards(parent.lchild, in_c)
      if not child_fixed: # allowed to broadcast so 1 is also a valid channel count
        child_fixed = fix_channel_count_downwards(parent.lchild, broadcast_width)
    if child_fixed:
      return fix_channel_count_upwards(parent, in_c)
    else:
      return False
  elif isinstance(parent, BinopIJK):
    if cur_node is parent.lchild:
      return fix_channel_count_upwards(parent, in_c + parent.rchild.out_c)
    else:
      return fix_channel_count_upwards(parent, in_c + parent.lchild.out_c)
  elif isinstance(parent, UnopIJ) or isinstance(parent, UnopI1):
    parent.in_c = in_c
    return True
  elif isinstance(parent, BinopIcJcKc) or isinstance(parent, Const):
    if cur_node is parent.lchild:
      return parent.in_c[0] == in_c
    else:
      return parent.in_c[1] == in_c
  elif isinstance(parent, TernaryHcIcJcKc):
    if cur_node is parent.child1:
      return parent.in_c[0] == in_c
    elif cur_node is parent.child2:
      return parent.in_c[1] == in_c
    else:
      return parent.in_c[2] == in_c
  elif isinstance(parent, UnopIcIc):
    return parent.in_c == in_c
  elif isinstance(parent, tuple):
    fixed = True
    for p in parent:
      fixed = fixed and fix_channel_count_upwards(p, in_c)
    return fixed
  else:
    return fix_channel_count_upwards(parent, in_c)
  

def fix_channel_count_upwards(subtree, in_c):
  if type(subtree.parent) is tuple:
    for p in subtree.parent:
      if not fix_channel_count_upwards_helper(subtree, p, in_c):
        return False
    return True
  else:
    return fix_channel_count_upwards_helper(subtree, subtree.parent, in_c)


"""
type check that linear operators are followed by nonlinear operators
"""
def check_linear_types(root):
  if isinstance(root, Linear):
    if root.num_children == 2:
      assert(isinstance(root.lchild, NonLinear) or isinstance(root.lchild, Special) \
              or isinstance(root.rchild, NonLinear) or isinstance(root.rchild, Special))
      check_linear_types(root.lchild)
      check_linear_types(root.rchild)
    elif root.num_children == 1:
      assert(isinstance(root.child, NonLinear) or isinstance(root.child, Special))
      check_linear_types(root.child)
  if isinstance(root, NonLinear):
    if root.num_children == 2:
      assert(isinstance(root.lchild, Linear) or isinstance(root.lchild, Special) \
              or isinstance(root.rchild, Linear) or isinstance(root.rchild, Special))
      check_linear_types(root.lchild)
      check_linear_types(root.rchild)
    elif root.num_children == 1:     
      assert(isinstance(root.child, Linear) or isinstance(root.child, Special))
      check_linear_types(root.child)
  else:
    if root.num_children == 4:
      check_linear_types(root.child1)
      check_linear_types(root.child2)
      check_linear_types(root.child3)
      check_linear_types(root.child4)
    elif root.num_children == 3:
      check_linear_types(root.child1)
      check_linear_types(root.child2)
      check_linear_types(root.child3)
    elif root.num_children == 2:
      check_linear_types(root.lchild)
      check_linear_types(root.rchild)
    elif root.num_children == 1:
      check_linear_types(root.child)
      

"""
returns if tree is linear - meaning only has Add, Sub, or Convs
"""
def is_linear(tree):
  tree_type = type(tree)
  if tree_type is Input:
    return True
  if tree_type is Add or tree_type is Sub or isinstance(tree, Linear):
    if tree.num_children == 4:
      return isinstance(tree, Linear) and is_linear(tree.child1) and \
              is_linear(tree.child2) and is_linear(tree.child3) and is_linear(tree.child4)
    elif tree.num_children == 3:
      return isinstance(tree, Linear) and is_linear(tree.child1) and is_linear(tree.child2) and is_linear(tree.child3)
    elif tree.num_children == 2:
      return isinstance(tree, Linear) and is_linear(tree.lchild) and is_linear(tree.rchild)
    elif tree.num_children == 1:
      return isinstance(tree, Linear) and is_linear(tree.child)
  return False

"""
counts the number of convolutional layers in a tree
"""
def count_parameterized_depth(tree):
  if tree.num_children == 4:
    child1_d = count_parameterized_depth(tree.child1)
    child2_d = count_parameterized_depth(tree.child2)
    child3_d = count_parameterized_depth(tree.child3)
    child4_d = count_parameterized_depth(tree.child4)
    depth = max([child1_d, child2_d, child3_d, child4_d])
  elif tree.num_children == 3:
    child1_d = count_parameterized_depth(tree.child1)
    child2_d = count_parameterized_depth(tree.child2)
    chidl3_d = count_parameterized_depth(tree.child3)
    depth = max([child1_d, child2_d, child3_d])
  elif tree.num_children == 2:
    lchild_d = count_parameterized_depth(tree.lchild)
    rchild_d = count_parameterized_depth(tree.rchild)
    depth = max(lchild_d, rchild_d)
  elif tree.num_children == 1:
    depth = count_parameterized_depth(tree.child)
  else:
    depth = 0
  if isinstance(tree, Linear):
    depth += 1

  return depth







