from demosaic_ast import *


"""
checks that channels counts of AST agree across nodes
and returns output channels of the root node
"""
def check_channel_count(node):
  if isinstance(node, BinopIII):
    lchild_c = check_channel_count(node.lchild)
    rchild_c = check_channel_count(node.rchild)
    assert(lchild_c == rchild_c or rchild_c == 1 or lchild_c == 1) # allow broadcasting
    return lchild_c
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
returns list of earliest occurence(s) of type in tree
and the level of occurence
"""
def find_type(root, T, level, ignore_root=False):
  if isinstance(root, T) and not ignore_root:
    return [root], level
  elif isinstance(root, Const): # reached leaf node
    return None, 1e10
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
    found, l = find_type(root.child, T, level+1)
    return found, l


"""
attempts to fix output channels of given ast to match out_c
by finding topmost UnopIJ nodes and changing their output channels to out_c
"""
def fix_channel_count_downwards(root, out_c):
  if isinstance(root, BinopIII):
    lfixed = fix_channel_count_downwards(root.lchild, out_c)
    if not lfixed:
      lfixed = fix_channel_count_downwards(root.lchild, 1)
    if lfixed:
      rfixed = fix_channel_count_downwards(root.rchild, out_c)
      if not rfixed:
        rfixed = fix_channel_count_downwards(root.rchild, 1)
      return rfixed
    else:
      return False
  elif isinstance(root, BinopIJK):
    # naively divide out_c evenly across inputs
    lfixed = fix_channel_count_downwards(root.lchild, out_c//2)
    if lfixed:
      rfixed = fix_channel_count_downwards(root.rchild, out_c//2)
      return rfixed
    else:
      return False
  elif isinstance(root, UnopIJ):
    root.out_c = out_c
    return True
  elif isinstance(root, BinopIcJcKc) or isinstance(root, UnopI1) or isinstance(root, Const):
    if root.out_c != out_c:
      return False
    else:
      return True
  else:
    return fix_channel_count_downwards(root.child, out_c)


"""
attempts to fix input channels of parent tree to match output channels of subtree
by finding closest UnopIJ ancestor and changing its input channels to out_c
if a Binop is encountered, fix the channel counts downwards for the other child
"""
def fix_channel_count_upwards_helper(subtree, parent, in_c):
  cur_node = subtree
  if parent is None:
    return True
  elif isinstance(parent, BinopIII):
    if cur_node is parent.lchild:
      child_fixed = fix_channel_count_downwards(parent.rchild, in_c)
      if not child_fixed: # allowed to broadcast so 1 is also a valid channel count
        child_fixed = fix_channel_count_downwards(parent.rchild, 1)
    else:
      child_fixed = fix_channel_count_downwards(parent.lchild, in_c)
      if not child_fixed: # allowed to broadcast so 1 is also a valid channel count
        child_fixed = fix_channel_count_downwards(parent.lchild, 1)
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
      if not (isinstance(root.child, NonLinear) or isinstance(root.child, Special)):
        print("root children should be NonLinear or Special: {}".format(root.dump()))
      assert(isinstance(root.child, NonLinear) or isinstance(root.child, Special))
      check_linear_types(root.child)
  if isinstance(root, NonLinear):
    if root.num_children == 2:
      assert(isinstance(root.lchild, Linear) or isinstance(root.lchild, Special) \
              or isinstance(root.rchild, Linear) or isinstance(root.rchild, Special))
      check_linear_types(root.lchild)
      check_linear_types(root.rchild)
    elif root.num_children == 1:
      if not (isinstance(root.child, Linear) or isinstance(root.child, Special)):
        print("root children should be Linear or Special: {}".format(root.dump()))
      
      assert(isinstance(root.child, Linear) or isinstance(root.child, Special))
      check_linear_types(root.child)
  else:
    if root.num_children == 2:
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
    if tree.num_children == 2:
      return is_linear(tree.lchild) and is_linear(tree.rchild)
    elif tree.num_children == 1:
      return is_linear(tree.child)
  return False

