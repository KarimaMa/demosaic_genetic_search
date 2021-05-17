from demosaic_ast import *
from tree_manipulation import get_children, replace_child, replace_parent

 
"""
finds any adjacent relus and removes one of them
"""
def fix_nonlinear_adjacency(root):
  if isinstance(root, Relu):
    if isinstance(root.child, Relu):
      grand_child = root.child.child
      replace_child(root, root.child, grand_child)
      replace_parent(grand_child, root.child, root)
  else:
    children = get_children(root)
    for c in children:
      fix_nonlinear_adjacency(c)