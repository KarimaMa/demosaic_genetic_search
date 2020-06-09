"""
mutation rules
we can mutate within program regions specified by the 
chroma region, green region, and touchup region
allowed to copy subtrees from any region A to any other
region B as long as 
1) all leaves in A are allowed inputs to region B
Specifically: 
    chroma region has inputs GreenExtractor, bayer
    green region has inputs bayer
    touchup region has inputs Stack(ChromaExtractor, GreenExtractor
2) input / output channels match up
"""
import random
import logging
import copy
from inspect import signature
from demosaic_ast import *
from type_check import *
from test_models import *
from tree import *

logger = logging.getLogger("DebugLogger")
logger.setLevel(logging.DEBUG)

CHANNELS = set((1,4,8,12))

"""
picks a random op in tree to exchange with another type of op
"""

def legal_parent_child_linearity(parent, child):
  return parent_type in nl_and_sp and child_type in l_and_sp \
      or parent_type in l_and_sp and child_type in nl_and_sp
  
"""
When deleting nodes in tree, reassign any subtrees that have dependees 
to one of their dependees
"""
def reassign_providers(deleted_nodes):
  for n in deleted_nodes:
    if hasattr(n, 'dependees'):
      for d in n.dependees:
        if not d in deleted_nodes:
          d.parent.child = n
          n.parent = d.parent
          n.dependees.remove(d)
          break
     
  
"""
deletes nodes starting from the given node up the tree until
linearity rules are obeyed.
reassigns deleted subtrees to one if any of its dependees
returns the child below the given node to delete
"""
def delete_nodes(node):
  parent = node.parent
  cur_node = node
  deleted_nodes = set()

  while True:
    deleted_nodes.add(cur_node)
    if issubclass(type(cur_node), Binop):
      # decide if you're going to keep left or right child
      flip = random.randint(0,1)
      if flip: 
        child = cur_node.lchild
        deleted_nodes.add(set(cur_node.rchild.preorder()))
      else:
        child = cur_node.rchild
        deleted_nodes.add(set(cur_node.lchild.preorder()))
    else: # deleting a Unop
       child = cur_node.child

    # remove the current node
    if parent is None:
      break
    if parent.num_children == 2:
      if cur_node is parent.lchild:
        parent.lchild = child
      else:
        parent.rchild = child 
    else:
      parent.child = child

    if legal_parent_child_linearity(parent, child):
      break
    
    cur_node = parent
    parent = cur_node.parent

  reassign_providers(deleted_nodes)
  return child

"""
picks a node to delete - ok as long as not input node
"""
def select_node_to_delete(node_list):
  while True:
    # delete the selected node
    node_id = random.randint(0, len(node_list)-1)
    node = preorder_nodes[node_id]
    if not isinstance(node, Input):
      break
  return node


"""
picks a randomm op in tree to delete
"""
def delete_mutation(tree):
  preorder_nodes = tree.preorder()
  tree.assign_parents()
  # delete the selected node
  node = select_node_to_delete(preorder_nodes)
  deletion_child = delete_nodes(node)
  deletion_parent = deletion_child.parent
  # check that channel counts are ok
  if deletion_parent.num_children == 2:
    if deletion_child is deletion_parent.lchild:
      out_c = deletion_parent.in_c[0]
    else:
      out_c = deletion_parent.in_c[1]
  else:
    out_c = deletion_parent.in_c

  # try to fix channels from child down
  child_copy = copy.deepcopy(deletion_child)
  fixed = fix_channel_count_downwards(child_copy, out_c)
  if fixed:
    if deletion_parent.num_children == 2:
      if deletion_child is deletion_parent.lchild:
        deletion_parent.lchild = child_copy
      else:
        deletion_parent.rchild = child_copy
    else:
      deletion_parent.child = child_copy
    tree.compute_input_output_channels()
    check_channel_count(tree)
  else:
    print("failed to fix channel counts downwards from deletion child")
    # try to fix channels from parent up
    deletion_child.compute_input_output_channels()
    fixed = fix_channel_count_upwards(deletion_child, deletion_child.out_c)
    if fixed:
      tree.compute_input_output_channels()
      check_channel_count(tree)

  return tree 


"""
Returns sets of valid op type(s) to insert given the types of the 
parent and child nodes: Linear, Special, or NonLinear.
May return one or two sets. If one set is returned, one node will be 
inserted with an op type belonging to the returned set.
If two sets are returned, two nodes will be inserted with their types
dictated by those two sets. 
"""
def get_insert_types(parent, child):
  if isinstance(parent, Linear):
    if isinstance(child, NonLinear):
      flip = random.randint(0,1)
      if flip:
        return [special_ops]
      else:
        return [nonlinear_ops, linear_ops]
    elif isinstance(child, Special):
      return [nl_and_sp]
    else:
      assert False, "Linear parent cannot have Linear child"
  elif isinstance(parent, NonLinear):
    if isinstance(child, Linear):
      flip = random.randint(0,1)
      if flip:
        return [special_ops]
      else:
        return [linear_ops, nonlinear_ops]
    elif isinstance(child, Special):
      return [l_and_sp]
    else:
      assert False, "NonLinear parent cannot have NonLinear child"
  elif isinstance(parent, Special):
    if isinstance(child, Linear):
      return [nl_and_sp]
    elif isinstance(child, NonLinear):
      return [l_and_sp]
    else:
      return [all_ops]


"""
Inserts nodes of the given insert ops above the insert child
in the order: insert_parent [nodes of insert ops] insert_child
tree: the tree to insert into
insert_ops: the ops to create new nodes from
insert_child: the node below the inserted nodes
input_set: the inputs that the resulting subtree is allowed to use

returns the new nodes in the same order as their corresponding insert_ops
"""
def insert(tree, insert_ops, insert_child, input_set):
  insert_parent = insert_child.parent
  print("inserting between parent {} and child {}".format(insert_parent.name, insert_child.name))
  print(insert_parent.dump())
  print("inserting ops: ")
  print(insert_ops)

  cur_child = insert_child
  new_nodes = [] 
  for i,OpClass in enumerate(reversed(insert_ops)):
    if issubclass(OpClass, Binop):
      params = list(signature(OpClass).parameters.items())
      assert len(params) == 2, "Invalid number of parameters for Binary op"

      # choose subtree as other child 
      weight_share = random.randint(0,1)
      print("is opclass a binopIII {}".format(issubclass(OpClass, BinopIII)))
      if issubclass(OpClass, BinopIII):
        # choose to weight share with subtree or copy its structure for child node
        if weight_share:
          subtree = pick_subtree(tree, input_set)
          subtree.compute_input_output_channels()         

          # make other child's output channels match subtree's output channels
          # requires dropping the other child and creating a copy of it so that dependees are not affected
          deleted_nodes = cur_child.preorder()          
          reassign_providers(deleted_nodes)
          cur_child = copy.deepcopy(cur_child)
          fixed = fix_channel_count_downwards(cur_child, subtree.out_c)
          print("weight sharing with subtree, make child output channels match {}".format(subtree.out_c))
          print("subtree")
          print(subtree.dump())
          print("child")
          print(cur_child.dump())
          if not fixed:
            print("unable to make binopIII's child node output channel count agree with subtree output channels")
            print("cur child")
            print(cur_child.dump())
            print("subtree")
            print(subtree.dump())
            assert False, "unable to make binopIII child node output channel count agree with subtree output channels"
        else: # not weight sharing with chosen subtree, copy it over
          # make subtree copy's output channels match other child's output channels
          cur_child.compute_input_output_channels()
          target_out_c = cur_child.out_c
          print("not weight sharing, make subtree output channels match child output channels {}".format(target_out_c))
          print("trying to find insert subtree with output channels {}".format(target_out_c))
          print("cur child")
          print(cur_child.dump())
          subtree = pick_subtree(tree, input_set, target_out_c)
          print("subtree")
          print(subtree.dump())

      else: # Op is BinopIJK
        print("insert op is a STACK")
        if weight_share: # don't modify chosen subtree output channels
          subtree = pick_subtree(tree, input_set)
          print("weight sharing with subtree, don't change its output channels")
        else: # randomly pick output channels 
          target_out_c = random.sample(CHANNELS, 1)[0]
          print("not weight sharing, make subtree output channels randomly chosen output channels {}".format(target_out_c))
          subtree = pick_subtree(tree, input_set, target_out_c)

      if weight_share:
        subtree.compute_input_output_channels()
        subtree_child = Input(subtree.out_c, node=subtree)
      else:
        subtree_child = subtree
      flip = random.randint(0,1)
      if flip:
        new_node = OpClass(cur_child, subtree_child)
      else:
        new_node = OpClass(subtree_child, cur_child)

    elif issubclass(OpClass, Unop):
      params = list(signature(OpClass).parameters.items())
      if len(params) == 2:
        out_c = random.sample(CHANNELS, 1)[0]
        new_node = OpClass(cur_child, out_c)
      elif len(params) == 1:
        new_node = OpClass(cur_child)
      else:
        assert False, "Invalid number of parameters for Unary op"

    new_nodes = [new_node] + new_nodes
    cur_child.parent = new_node
    cur_child = new_node
    cur_child.compute_input_output_channels()

  cur_child.parent = insert_parent
  if insert_parent.num_children == 2:
    if insert_child is insert_parent.rchild:
      insert_parent.rchild = cur_child
    else:
      insert_parent.lchild = cur_child
  elif insert_parent.num_children == 1:
    insert_parent.child = cur_child
  else:
    assert False, "Should not be inserting after an input node"

  # make sure output channels of inserted nodes agree with parent
  # go back up to 
  cur_child.compute_input_output_channels()
  fixed = fix_channel_count_upwards(cur_child, cur_child.out_c)
  if not fixed:
    print("unable to make inserted nodes channel counts agree with tree")
    print("cur child")
    print(cur_child.dump())
    print("tree")
    print(tree.dump())
  else:
    print("fixed channel counts upwards")
    print("cur child")
    print(cur_child.dump())
    print("tree")
    print(tree.dump())

  tree.compute_input_output_channels()
  check_channel_count(tree)
  tree.add_dependees()

  return new_nodes

"""
Trys to mutate tree by inserting random op(s) at a random location
If op is a binary op, picks a subtree to add as the other child
"""
def insert_mutation(tree, input_set):
  preorder_nodes = tree.preorder()
  tree.assign_parents()
  # insert above the selected node
  insert_above_node_id = random.randint(1, len(preorder_nodes)-1)
  insert_child = preorder_nodes[insert_above_node_id]
  insert_parent = insert_child.parent

  # pick op(s) to insert
  insert_types = get_insert_types(insert_parent, insert_child)
  while True:
    new_ops = []
    for n in range(len(insert_types)):
      OpClass = random.sample(insert_types[n], 1)[0]
      new_ops += [OpClass]
    if sum(map(lambda x : x in sandwich_ops, new_ops)) <= 1: 
      # allow at most one sandwich op to be inserted
      break
      
  new_nodes = insert(tree, new_ops, insert_child, input_set)
  # if we inserted a sandwich op, we must insert its partner node as well
  for OpClass, new_node in zip(new_ops, new_nodes):
    if OpClass in sandwich_ops:
      OpPartner = sandwich_pairs[OpClass]  
      # decide where to insert partner node, for now partners are always Unops
      ancestors_from_binop_to_node = find_closest_subclass_ancestor(new_node, Binop)
      # don't insert partner right above node - makes sandwich pointless
      insert_above_node_id = random.randint(1, max(len(ancestors_from_binop_to_node)-2, 1))
      insert_child = ancestors_from_binop_to_node[insert_above_node_id]
      partner_node = insert(tree, [OpPartner], insert_child, input_set)
        
  return tree

"""
checks whether subtree obeys the restricted input set and can be modified 
to agree with given output channels.
Returns a copy of the subtree with channels appropriately modified
"""
def allow_subtree(root, input_set, target_out_c=None):
  size = root.compute_size(set(), count_input_exprs=True)
  # reject small trees
  if size < 2 or size > 10: 
      print("Subtree of size {} is too small or too large".format(size))
      assert False, "Subtree of size {} is too small or too large".format(size)

  # reject trees that require inputs out of sanctioned input_set
  subtree_inputs = root.get_inputs()
  if not subtree_inputs.issubset(input_set):
    print("rejecting subtree with invalid inputs")
    assert False, "rejecting subtree with invalid inputs"

  # reject trees that can't be made to aggree with desired output channels
  if target_out_c:
    root_copy = copy.deepcopy(root)
    in_c, out_c = root_copy.compute_input_output_channels()
    if out_c != target_out_c: 
      # copy the subtree in case changes made by attempt to fix the channels doesn't work
      fixed = fix_channel_count_downwards(root_copy, target_out_c)
      if fixed:
        in_c, out_c = root_copy.compute_input_output_channels()
      else:
        print("rejecting subtree: cannot make output channels {} match {}".format(out_c, target_out_c))
        print(root_copy.dump())
        assert False, "rejecting subtree: cannot make output channels {} match {}".format(out_c, target_out_c)
    return root_copy
  else:
    return root
  

def pick_subtree(root, input_set, target_out_c=None):
  preorder = full_model.preorder()
  while True:
    subtree_id = random.randint(0, len(preorder)-1)
    subtree = preorder[subtree_id]
    in_c, out_c = subtree.compute_input_output_channels()
    try:
      subtree_copy = allow_subtree(subtree, input_set, target_out_c)
    except AssertionError:
      print("selected subtree is invalid")
    else:
      print("successfully selected subtree for insertion")
      print("---selected subtree---")
      print(subtree_copy.dump())
      return subtree_copy


#TODO: CHECK THAT LINEARITY TYPE CHECKS
if __name__ == "__main__":

  bayer = Input(1, "Bayer")
  green_model = build_green_model(bayer)
  full_model, inputs = build_full_model(green_model, bayer)
  print("full model inputs")
  print(inputs)
  derived_inputs = full_model.get_inputs()
  treestr = full_model.dump()
  print(treestr)

  check_linear_types(full_model)
  allowed_inputs = inputs
  print("--------------- MUTATING TREE ------------------")
  new_model = insert_mutation(full_model, allowed_inputs)
  check_linear_types(new_model)
  print("the new model after insert mutation")
  print(new_model.dump())

