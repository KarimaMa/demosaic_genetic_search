from demosaic_ast import *
from util import get_closest_factor, get_factors 
from enum import Enum 
import logging
import math
 

class Resolution(Enum):
  FULL = 1
  DOWNSAMPLED = 2
  INVALID = 3


"""
returns max channel over all nodes in DAG
"""
def get_max_channels(node):
  max_channels = node.out_c
  if node.num_children == 3:
    children = [node.child1, node.child2, node.child3]
  elif node.num_children == 2:
    children = [node.lchild, node.rchild]
  elif node.num_children == 1:
    children = [node.child]
  else:
    return max_channels
  child_channels = [get_max_channels(c) for c in children]
  max_channels = max(max_channels, max(child_channels))
  return max_channels


"""
only shrink channels if channel count exceeds a max threshold
"""
def shrink_channels(node, max_c, out_c=None):
  if isinstance(node, BinopIJK):
    if out_c is None:
      left_in_c = shrink_channels(node.lchild, max_c)
      right_in_c = shrink_channels(node.rchild, max_c)
      node.in_c = (left_in_c, right_in_c)
      node.out_c = left_in_c + right_in_c
    else:
      assert out_c == node.out_c, "cannot force stack to change output channels"
      left_in_c = node.lchild.out_c
      right_in_c = node.rchild.out_c
      shrink_channels(node.lchild, max_c, out_c=left_in_c)
      shrink_channels(node.rchild, max_c, out_c=right_in_c)
    return node.out_c
  elif isinstance(node, BinopIII):   
    if out_c is None:
      left_in_c = shrink_channels(node.lchild, max_c) 
      right_in_c = shrink_channels(node.rchild, max_c)
      assert left_in_c == right_in_c or (left_in_c % right_in_c == 0) or (right_in_c % left_in_c == 0), \
        f"Shrinking channels cannot make BinopIII left {left_in_c} match right {right_in_c} channels"
      node.out_c = max(left_in_c, right_in_c)
      node.in_c = (left_in_c, right_in_c)
      return node.out_c
    else:
      try: # trying (out_c, out_c)
        shrink_channels(node.lchild, max_c, out_c=out_c)
        shrink_channels(node.rchild, max_c, out_c=out_c)
      except AssertionError:
        assert False, f"Failed to make BinopIII output channels agree with {out_c}"
      else:
        node.in_c = (out_c, out_c)
        node.out_c = out_c
        return node.out_c
  elif isinstance(node, BinopIcJcKc): # input and output channels are immutable
    if not out_c is None:
      assert out_c == node.Kc(), f"output channels of {node.__class__.__name__} cannot be set to {out_c}"
    shrink_channels(node.lchild, max_c, out_c=node.Ic())
    shrink_channels(node.rchild, max_c, out_c=node.Jc())
    return node.out_c
  elif isinstance(node, TernaryHcIcJcKc):
    if not out_c is None:
      assert out_c == node.Kc(), f"output channels of {node.__class__.__name__} cannot be set to {out_c}"
    shrink_channels(node.child1, max_c, out_c=node.Hc())
    shrink_channels(node.child2, max_c, out_c=node.Ic())
    shrink_channels(node.child3, max_c, out_c=node.Jc()) 
    return node.out_c  
  elif isinstance(node, UnopIcJc): # Flat2Quad or GreenRBExtractor
    if not out_c is None:
      assert out_c == node.Jc(), f"output channels of {node.__class__.__name__} cannot be set to {out_c}"
    shrink_channels(node.child, max_c, out_c=node.Ic())
    return node.out_c
  elif isinstance(node, UnopIJ):
    if out_c is None:
      node.out_c = min(node.out_c, max_c)
      node.in_c = shrink_channels(node.child, max_c)
    else:
      node.out_c = out_c
      node.in_c = shrink_channels(node.child, max_c)
    return node.out_c
  elif isinstance(node, UnopII):
    if out_c is None:
      node.in_c = shrink_channels(node.child, max_c)
      node.out_c = node.in_c
    else:
      shrink_channels(node.child, max_c, out_c=out_c)
      node.out_c = out_c
      node.in_c = out_c
    return node.out_c
  elif isinstance(node, UnopIIdiv):
    if out_c is None:
      max_input_c = (max_c // node.out_c) * node.out_c
      if max_input_c < node.in_c:
        node.in_c = shrink_channels(node.child, max_c, out_c=max_input_c) 
      else:
        shrink_channels(node.child, max_c, out_c=node.in_c) 
    else:
      # set input channels to closest multiple of out_c to current in_c 
      in_c = math.ceil(node.in_c / out_c) * out_c
      shrink_channels(node.child, max_c, out_c=in_c)
      node.in_c = in_c
      node.out_c = out_c
    return node.out_c
  elif isinstance(node, Const):
    if not out_c is None:
      assert node.out_c == out_c, f"Cannot set output channels of {node.name} to {out_c}"
    return node.out_c 
  else:
    assert False, "Unknown node type"


"""
returns none if resolution is invalid
"""
def compute_resolution(tree):
  if isinstance(tree, Input):
    return tree.resolution
  elif isinstance(tree, LearnedDownsample) or isinstance(tree, Pack):
    child_resolution = compute_resolution(tree.child)
    tree.resolution = child_resolution / tree.factor 
  elif isinstance(tree, Upsample) or isinstance(tree, LearnedUpsample) or isinstance(tree, Unpack):
    child_resolution = compute_resolution(tree.child)
    tree.resolution = child_resolution * tree.factor
  else:
    if tree.num_children == 3:
      res1 = compute_resolution(tree.child1)
      res2 = compute_resolution(tree.child2)
      res3 = compute_resolution(tree.child3)
      if res1 != res2 or res1 != res3:
        return None
      tree.resolution = res1
    elif tree.num_children == 2:
      lres = compute_resolution(tree.lchild)
      rres = compute_resolution(tree.rchild)
      if lres != rres:
        return None
      tree.resolution = lres
    elif tree.num_children == 1:
      child_res = compute_resolution(tree.child)
      tree.resolution = child_res
  
  return tree.resolution


"""
returns the spatial resolution of a subtree
"""
def spatial_resolution(subtree):
  res = None # resolution obeys children unless this node is an up or downsample
  if isinstance(subtree, Upsample):
    res = Resolution.FULL
  elif isinstance(subtree, Downsample):
    res = Resolution.DOWNSAMPLED

  if subtree.num_children == 3:
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
    lchild_c = check_channel_count(node.lchild)
    rchild_c = check_channel_count(node.rchild)
    ok = lchild_c == rchild_c or rchild_c % lchild_c == 0 or lchild_c % rchild_c == 0
    assert ok, f"BinopIII {node.dump()}" # allow broadcasting
    return max(rchild_c, lchild_c)
  elif isinstance(node, BinopIJK):
    lchild_c = check_channel_count(node.lchild)
    rchild_c = check_channel_count(node.rchild)
    out_c = lchild_c + rchild_c
    return out_c
  elif isinstance(node, BinopIcJcKc):
    lchild_c = check_channel_count(node.lchild)
    rchild_c = check_channel_count(node.rchild)
    assert(lchild_c == node.Ic() and rchild_c == node.Jc()), f"BinopIcJcKc {node.dump()}" 
    return node.Kc()
  elif isinstance(node, TernaryHcIcJcKc):
    child1_c = check_channel_count(node.child1)
    child2_c = check_channel_count(node.child2)
    child3_c = check_channel_count(node.child3)
    assert(child1_c == node.Hc() and child2_c == node.Ic() and child3_c == node.Jc())
    return node.Kc()
  elif isinstance(node, UnopIcJc):
    child_c = check_channel_count(node.child)
    assert(child_c == node.Ic()), f"UnopIcJc {node.dump()}" 
    return node.Jc()
  elif isinstance(node, UnopII):
    child_out_c = check_channel_count(node.child)
    assert(node.in_c == child_out_c), f"UnopII {node.dump()}" 
    return node.out_c
  elif isinstance(node, UnopIJ):
    assert((node.in_c % node.groups == 0) and (node.out_c % node.groups == 0)), f"UnopIJJ {node.dump()}"
    child_out_c = check_channel_count(node.child)
    assert(node.in_c == child_out_c), f"UnopIJ {node.dump()} "
    return node.out_c
  elif isinstance(node, UnopIIdiv):
    assert((node.in_c % node.out_c == 0) and (node.in_c >= node.out_c)), f"UnopIIdiv {node.dump()}"
    child_out_c = check_channel_count(node.child)
    assert (node.in_c == child_out_c), f"UnopIIdiv {node.dump()}"
    return node.out_c
  elif isinstance(node, UnopIJFixed):
    if isinstance(node, Unpack):
      assert( (node.in_c / node.factor**2) == node.out_c ), f"Unpack {node.dump()}"
    elif isinstance(node, Pack):
      assert( (node.in_c * node.factor**2) == node.out_c ), f"Pack {node.dump()}"
    child_out_c = check_channel_count(node.child)
    assert(node.in_c == child_out_c)
    return node.out_c
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
def fix_channel_count_downwards(root, parent, out_c, fixed_nodes=None):
  if fixed_nodes is None:
    fixed_nodes = {}

  if id(root) in fixed_nodes: # can't change channels - check it matches required out_c
    fixed_out_c = fixed_nodes[id(root)].out_c

    if fixed_out_c != out_c:
      return False
    return True
  else: # node is seen for the first time
    fixed = False
    
    fixed_nodes[id(root)] = root

    if isinstance(root, BinopIII):
      lfixed = False
      rfixed = False
      lfixed = fix_channel_count_downwards(root.lchild, root, out_c, fixed_nodes)
      if lfixed:
        root.lchild.compute_input_output_channels()
        if (root.lchild.out_c % root.rchild.out_c == 0):
          root.in_c = (root.lchild.out_c, root.rchild.out_c)
          root.out_c = out_c
          fixed = True
        else: # try to make rchild also have out_c output channels
          fixed = fix_channel_count_downwards(root.rchild, root, out_c, fixed_nodes)
          if fixed:
            root.rchild.compute_input_output_channels()
            root.in_c = (out_c, out_c)
            root.out_c = out_c
      else:
        rfixed = fix_channel_count_downwards(root.rchild, root, out_c, fixed_nodes)
        if rfixed:
          root.rchild.compute_input_output_channels()
          if (root.rchild.out_c % root.lchild.out_c == 0):
            root.in_c = (root.lchild.out_c, root.rchild.out_c)
            root.out_c = out_c
            fixed = True
        fixed = False
    elif isinstance(root, BinopIJK):
      lchild_out_c = out_c // 2
      rchild_out_c = out_c - lchild_out_c
      if lchild_out_c == 0:
        fixed = False
      else: # naively divide out_c evenly across inputs
        lfixed = fix_channel_count_downwards(root.lchild, root, lchild_out_c, fixed_nodes)
        if lfixed:
          rfixed = fix_channel_count_downwards(root.rchild, root, rchild_out_c, fixed_nodes)
          fixed = rfixed
        else:
          fixed = False
    elif isinstance(root, UnopIJ):
      root.out_c = out_c
      # change grouping if old groups is no longer divisible by the new input channels
      in_c_factors = get_factors(root.in_c)
      if type(root) is Conv1D:
        out_c_factors = get_factors(root.out_c // 2)
      else:
        out_c_factors = get_factors(root.out_c)
      factors = in_c_factors.intersection(out_c_factors)
      closest_factor = get_closest_factor(factors, root.groups)
      root.groups = closest_factor # if current groups is already a factor, this does nothing
      fixed = True
    elif isinstance(root, UnopIIdiv):
      if root.in_c % out_c == 0:
        root.out_c = out_c
        fixed = True
      else: # make input channels the closest multiple of out_c to current in_c
        in_c = math.ceil(root.in_c / out_c) * out_c       
        fixed = fix_channel_count_downwards(root.child, root, in_c, fixed_nodes)
        if fixed:
          root.in_c = in_c
          root.out_c = out_c
    elif isinstance(root, UnopIJFixed):
      # cannot change output channels, can only change input channels
      if root.out_c == out_c:
        fixed = True
      else:
        if isinstance(root, Pack):
          if out_c % (root.factor**2) == 0:
            in_c = int(out_c // (root.factor**2))
            fixed = fix_channel_count_downwards(root.child, root, in_c, fixed_nodes)
        elif isinstance(root, Unpack):
          in_c = out_c * int(root.factor**2)
          fixed = fix_channel_count_downwards(root.child, root, in_c, fixed_nodes)
    elif isinstance(root, TernaryHcIcJcKc) \
      or isinstance(root, BinopIcJcKc) \
      or isinstance(root, UnopIcJc) \
      or isinstance(root, Const):
      fixed = (root.out_c == out_c) 
    elif isinstance(root, UnopII): # is type UnopII
      fixed = fix_channel_count_downwards(root.child, root, out_c, fixed_nodes)
    else:
      assert False, "Unknown root type in fix channels downwards"
    # if fixed:
    #   fixed_nodes[id(root)] = root

    if type(root.parent) is tuple:
      for p in root.parent:
        if not id(p) == id(parent):
          fixed = fixed and fix_channel_count_upwards_helper(root, p, out_c, fixed_nodes)

    if not fixed:
      print(f"in fix downwards {id(root)} ")
    return fixed


def get_in_c_from_child(child, parent):
  if parent.num_children == 3:
    children = [parent.child1, parent.child2, parent.child3]
    in_c = [parent.in_c[0], parent.in_c[1], parent.in_c[2]]
  elif parent.num_children == 2:
    children = [parent.lchild, parent.rchild]
    in_c = [parent.in_c[0], parent.in_c[1]]
  else:
    children = [parent.child]
    in_c = [parent.in_c]
  for i, c in enumerate(children):
    if id(c) == id(child):
      return in_c[i]


"""
attempts to fix input channels of parent tree to match output channels of subtree
by finding closest UnopIJ ancestor and changing its input channels to out_c
if a Binop is encountered, fix the channel counts downwards for the other child
if the requried input channels is not 1.
"""
def fix_channel_count_upwards_helper(subtree, parent, in_c, fixed_nodes=None):
  cur_node = subtree
  if parent is None:
    return True

  if fixed_nodes is None:
    fixed_nodes = {}

  if id(parent) in fixed_nodes: # can't change channels - check it matches required out_c
    fixed_in_c = get_in_c_from_child(subtree, fixed_nodes[id(parent)])
    if fixed_in_c != in_c:
      return False
    return True

  fixed = True
  
  fixed_nodes[id(parent)] = parent
  
  if isinstance(parent, BinopIII):
    if cur_node is parent.lchild:
      parent.in_c = (in_c, in_c)
      # need to call fix upwards for all parents of child we fix downwards from 
      # in case we fix down a child with other parents than the one we are calling fix down from
      if parent.rchild.out_c != in_c:
        child_fixed = fix_channel_count_downwards(parent.rchild, parent, in_c, fixed_nodes)
      else:
        child_fixed = True
      if child_fixed:
        if type(parent.rchild.parent) is tuple:
          for p in parent.rchild.parent:
            if not id(p) == id(parent):
              child_fixed = child_fixed and fix_channel_count_upwards_helper(parent.rchild, p, in_c, fixed_nodes)

    else:
      parent.in_c = (in_c, in_c)
      if parent.lchild.out_c != in_c:
        child_fixed = fix_channel_count_downwards(parent.lchild, parent, in_c, fixed_nodes)
      else:
        child_fixed = True
      if child_fixed:
        if type(parent.lchild.parent) is tuple:
          for p in parent.lchild.parent:
            if not id(p) == id(parent):
              child_fixed = child_fixed and fix_channel_count_upwards_helper(parent.lchild, p, in_c, fixed_nodes) 
    if child_fixed:
      fixed = fix_channel_count_upwards(parent, in_c, fixed_nodes)
    else:
      fixed = False
  elif isinstance(parent, BinopIJK):
    if cur_node is parent.lchild:
      fixed = fix_channel_count_upwards(parent, in_c + parent.rchild.out_c, fixed_nodes)
    else:
      fixed = fix_channel_count_upwards(parent, in_c + parent.lchild.out_c, fixed_nodes)
  elif isinstance(parent, UnopIJ):
    parent.in_c = in_c
    # change grouping if old groups is no longer divisible by the new input channels
    in_c_factors = get_factors(parent.in_c)
    if type(parent) is Conv1D:
      out_c_factors = get_factors(parent.out_c // 2)
    else:
      out_c_factors = get_factors(parent.out_c)
    factors = in_c_factors.intersection(out_c_factors)
    closest_factor = get_closest_factor(factors, parent.groups)
    parent.groups = closest_factor # if current groups is already a factor, this does nothing

    fixed = True
  elif isinstance(parent, UnopIIdiv): # want parent to have in_c input channels 
    if in_c % parent.out_c == 0 and in_c >= parent.out_c:
      parent.in_c = in_c 
      fixed = True
    else:
      # find closest factor of in_c to current out_c of parent 
      in_c_factors = get_factors(in_c)
      closest_factor = get_closest_factor(in_c_factors, parent.out_c)
      fixed = fix_channel_count_upwards(parent, closest_factor, fixed_nodes)
      if fixed:
        parent.out_c = closest_factor
        parent.in_c = in_c
  elif isinstance(parent, UnopIJFixed): # want parent to have in_c input channels
    if in_c == parent.in_c:
      out_c = parent.out_c
      fixed = True
    elif isinstance(parent, Unpack):
      if in_c % (parent.factor**2) == 0:
        out_c = int(in_c // (parent.factor**2))
        fixed = fix_channel_count_upwards(parent, out_c, fixed_nodes)
      else:
        fixed = False
    elif isinstance(parent, Pack):
      out_c = in_c * int(parent.factor**2)
      fixed = fix_channel_count_upwards(parent, out_c, fixed_nodes)
    if fixed:
      parent.out_c = out_c 
      parent.in_c = in_c
  elif isinstance(parent, BinopIcJcKc):
    if cur_node is parent.lchild:
      fixed = parent.in_c[0] == in_c
    else:
      fixed = parent.in_c[1] == in_c
  elif isinstance(parent, TernaryHcIcJcKc):
    if cur_node is parent.child1:
      fixed = parent.in_c[0] == in_c
    elif cur_node is parent.child2:
      fixed = parent.in_c[1] == in_c
    else:
      fixed = parent.in_c[2] == in_c
  elif isinstance(parent, UnopIcJc):
    fixed = parent.in_c == in_c
  elif isinstance(parent, tuple):
    assert False, "PARENT SHOULD NEVER BE TUPLE WHEN CALLING FIX UPWARDS HELPER"
  else: # type is UnopII
    parent.in_c = in_c
    fixed = fix_channel_count_upwards(parent, in_c, fixed_nodes)

  return fixed


def fix_channel_count_upwards(subtree, in_c, fixed_nodes=None):
  if fixed_nodes is None:
    fixed_nodes = {}
  if type(subtree.parent) is tuple:
    for p in subtree.parent:
      if not fix_channel_count_upwards_helper(subtree, p, in_c, fixed_nodes):
        return False
    return True
  else:
    return fix_channel_count_upwards_helper(subtree, subtree.parent, in_c, fixed_nodes)


"""
checks that there are no two adjacent non-linear types
"""
def assert_no_nonlinear_adjacency(root):
  if isinstance(root, NonLinear):
    # has only one child
    assert(not isinstance(root.child, NonLinear))
    assert_no_nonlinear_adjacency(root.child)
  else:
    if root.num_children == 3:
      assert_no_nonlinear_adjacency(root.child1)
      assert_no_nonlinear_adjacency(root.child2)
      assert_no_nonlinear_adjacency(root.child3)
    elif root.num_children == 2:
      assert_no_nonlinear_adjacency(root.lchild)
      assert_no_nonlinear_adjacency(root.rchild)
    elif root.num_children == 1:
      assert_no_nonlinear_adjacency(root.child)


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
    if root.num_children == 3:
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
    if tree.num_children == 3:
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
  if tree.num_children == 3:
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


