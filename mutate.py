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
import time
import random
import logging
import copy
from inspect import signature
import numpy as np
from demosaic_ast import *
from type_check import *
from tree import *
from restrictions import *

LOG_FILENAME='log.txt'
logger = logging.getLogger("DebugLogger")
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

CHANNELS = set((1,4,8,12))
CHANNELS = set((8,))

"""
picks a random op in tree to exchange with another type of op
"""

def legal_parent_child_linearity(parent, child):
  if type(parent) is tuple:
    assert type(parent[0]) is type(parent[1]), "If a node has multiple parents they must have same type"
    parent_type = type(parent[0])
  else:
    parent_type = type(parent)
  child_type = type(child)
  logging.debug("parent type {} child type {}".format(parent_type, child_type))
  return parent_type in border_ops \
      or child_type in border_ops \
      or (parent_type in nl_and_sp and child_type in l_and_sp) \
      or (parent_type in l_and_sp and child_type in nl_and_sp)
  
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
given set of nodes, return all of their partner nodes
"""  
def get_partner_nodes(nodes):
  partner_nodes = set()
  for n in nodes:
    if hasattr(n, "partner"):
      partner_nodes.add(n.partner)
  return partner_nodes

"""
given a set of nodes, remove their connections to their parters
"""
def remove_partner_connections(nodes):
  for n in nodes:
    delattr(n, "partner")

"""
fixes up parent child relationships to remove given node
"""
def remove_node(parent, deletion_node, new_child):
  if parent.num_children == 2:
    if deletion_node is parent.lchild:
      parent.lchild = new_child
    else:
      parent.rchild = new_child 
  else:
    parent.child = new_child
  new_child.parent = parent


"""
fixes channel counts after deleting nodes
"""
def fix_channels_after_deletion(tree, deletion_parent, deletion_child):
  if deletion_parent is None:
    return 
  # check that channel counts are ok
  if deletion_parent.num_children == 2:
    if deletion_child is deletion_parent.lchild:
      out_c = deletion_parent.in_c[0]
    else:
      out_c = deletion_parent.in_c[1]
  else:
    out_c = deletion_parent.in_c

  # try to fix channels from child down
  # make a copy in case fixing fails and we need to try fixing upwards
  child_copy = copy_subtree(deletion_child)
  fixed = fix_channel_count_downwards(child_copy, out_c)
  if fixed:
    if deletion_parent.num_children == 2:
      if deletion_child is deletion_parent.lchild:
        deletion_parent.lchild = child_copy
      else:
        deletion_parent.rchild = child_copy
    else:
      deletion_parent.child = child_copy

    child_copy.parent = deletion_parent

    tree.compute_input_output_channels()
    check_channel_count(tree)
    logging.debug("fixed channel count downwards, child {}".format(child_copy.dump()))
  else:
    logging.debug("failed to fix channel counts downwards from deletion child {}".format(child_copy.dump()))
    # try to fix channels from parent up
    deletion_child.compute_input_output_channels()
    fixed = fix_channel_count_upwards(deletion_child, deletion_child.out_c)
    if fixed:
      tree.compute_input_output_channels()
      check_channel_count(tree)
      logging.debug("fixed channel count upwards, parent {}".format(deletion_child.parent.dump()))
      logging.debug("is child copy parent the deletion parent {}".format(deletion_child.parent is deletion_parent))
    else:
      logging.debug("failed to fix channel counts upwards from deletion child {}".format(deletion_child.dump()))
      assert False, "Could not make channel counts agree after deleting ops"


"""
deletes nodes from tree starting from the given node up the tree until
linearity rules are obeyed.
reassigns deleted subtrees to one if any of its dependees
returns the child of the deleted nodes
"""
def delete_nodes(tree, node):
  parent = node.parent
  cur_node = node
  deleted_nodes = set()

  while True:
    deleted_nodes.add(cur_node)
    logging.debug("deleting node {} node parent: {}".format(cur_node.name, cur_node.parent))
    if isinstance(cur_node, Binop):
      # if op is LogSub or AddExp always keep left child
      if type(cur_node) is LogSub or type(cur_node) is AddExp:
        keep_left = True
      else:
        # decide if you're going to keep left or right child
        keep_left = random.randint(0,1)
      if keep_left: 
        child = cur_node.lchild
        deleted_nodes = deleted_nodes.union(set(cur_node.rchild.preorder()))
      else:
        child = cur_node.rchild
        deleted_nodes = deleted_nodes.union(set(cur_node.lchild.preorder()))
    else: # deleting a Unop
       child = cur_node.child

    # remove the current node
    if parent is None:
      break

    if type(parent) is tuple:
      for p in parent:
        remove_node(p, cur_node, child)
    else:
      logging.debug("before removal: parent tree {}".format(parent.dump()))
      remove_node(parent, cur_node, child)
      logging.debug("after removal: parent tree {}".format(parent.dump()))
      logging.debug("after removal: entire tree {}".format(tree.dump()))

    if legal_parent_child_linearity(parent, child):
      logging.debug("legal parent child linearity")
      break
    else:
      logging.debug("not legal parent child linearity")
    # deletion caused illegal linearity, move up the tree to delete next node
    cur_node = parent
    if type(parent) is tuple: 
      # deleting LogSub or AddExp next - the other parent will be deleted by the 
      # code below that handles deleting partners
      parent = cur_node.parent[0]
    else:
      parent = cur_node.parent

  fix_channels_after_deletion(tree, parent, child)
  # delete partner if we deleted a sandwich node
  partners_of_deleted = get_partner_nodes(deleted_nodes)
  # must remove partner connections that refer back to deleted_nodes - else infinite recursion deletion
  remove_partner_connections(partners_of_deleted)
  for n in partners_of_deleted:
    delete_nodes(tree, n)
  
  return child


"""
picks a node to delete from a tree - ok as long as not a border node
rejects nodes for deletion based on how many nodes would be removed at most
for example, if deleting a binary node, we could remove as any nodes as its
largest subtree
"""
def select_node_to_delete(tree):
  preorder_nodes = tree.preorder()
  n = len(preorder_nodes)
  tree.compute_size(set(), count_all_inputs=True)
  while True:
    # delete the selected node
    node_id = random.randint(0, n-1)
    node = preorder_nodes[node_id]
    # reject deleting a node with c/n probability where n is total number
    # of nodes and c is the max number of nodes lost by deleting this node
    if isinstance(node, Binop):
      p_reject = float(max(node.lchild.size, node.rchild.size))/n
    else:
      p_reject = 1.0/n

    if random.uniform(0,1) < p_reject:
      continue
    if not (type(node) in border_ops):
      break
  return node


"""
picks a randomm op in tree to delete
"""
def delete_mutation(tree):
  # delete the selected node
  node = select_node_to_delete(tree)
  logging.debug("deleting node {}".format(node.dump()))
  logging.debug("from tree {}".format(tree.dump()))
  deletion_child = delete_nodes(tree, node)
  logging.debug("the tree after deletion {}".format(tree.dump()))

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
  if type(parent) is tuple:
    assert isinstance(parent[0], Linear) and isinstance(parent[1], Linear) or\
          isinstance(parent[0], NonLinear) and isinstance(parent[1], NonLinear) or\
          isinstance(parent[0], Special) and isinstance(parent[1], Special)
    parent = parent[0]
  if isinstance(parent, Linear):
    if isinstance(child, NonLinear):
      flip = random.randint(0,9)
      if flip < 7:
        return [special_insert_ops]
      else:
        return [nonlinear_insert_ops, linear_insert_ops]
    elif isinstance(child, Special):
      return [nl_sp_insert_ops]
    else:
      assert False, "Linear parent cannot have Linear child"
  elif isinstance(parent, NonLinear):
    if isinstance(child, Linear):
      flip = random.randint(0,9)
      if flip < 7:
        return [special_insert_ops]
      else:
        return [linear_insert_ops, nonlinear_insert_ops]
    elif isinstance(child, Special):
      return [l_sp_insert_ops]
    else:
      assert False, "NonLinear parent cannot have NonLinear child"
  elif isinstance(parent, Special):
    if isinstance(child, Linear):
      return [nl_sp_insert_ops]
    elif isinstance(child, NonLinear):
      return [l_sp_insert_ops]
    else:
      return [all_insert_ops]

"""
connections insertion parent node to newly created child
insert_child: the node above which the insertion occured
insert_parent: the parent of insert_child
node: the node inserted directly below insert_parent
"""
def link_insertion_parent(insert_parent, insert_child, node):
  if insert_parent.num_children == 2:
    if insert_child is insert_parent.rchild:
      insert_parent.rchild = node
    else:
      insert_parent.lchild = node
  elif insert_parent.num_children == 1:
    insert_parent.child = node
  else:
    assert False, "Should not be inserting after an input node"

"""
Inserts nodes of the given insert ops above the insert child
in the order: insert_parent [nodes of insert ops] insert_child
tree: the tree to insert into
insert_ops: tuples of: the op to create a new node from, the 
            subtree to use for other child if the op is a binary op, 
            and whether to use given subtree as the right child 
insert_child: the node below the inserted nodes
input_set: the inputs that the resulting subtree is allowed to use

returns the new nodes in the same order as their corresponding insert_ops
"""
def insert(tree, insert_ops, insert_child, input_set):
  insert_parent = insert_child.parent
  logging.debug("inserting above {}".format(insert_child.dump()))
  if insert_parent:
    if type(insert_parent) is tuple:  
      logging.debug("inserting {} between parent {} and child {}".format(insert_ops, (insert_parent[0].name, insert_parent[1].name), insert_child.name))
    else:
      logging.debug("inserting {} between parent {} and child {}".format(insert_ops, insert_parent.name, insert_child.name))

  cur_child = insert_child
  new_nodes = [] 
  for i, (OpClass, use_child, use_right) in enumerate(reversed(insert_ops)):
    if issubclass(OpClass, Binop):
      params = list(signature(OpClass).parameters.items())
      assert len(params) == 2, "Invalid number of parameters for Binary op"

      # choose subtree as other child 
      if issubclass(OpClass, BinopIII):
        # make subtree's output channels match other child's output channels
        cur_child.compute_input_output_channels()
        if use_child is None:
          subtree = pick_subtree(tree, input_set, target_out_c=cur_child.out_c)
          subtree.compute_input_output_channels()         
        else:
          subtree = use_child
          # make channel count of insert_child agree with required use_child
          fixed = fix_channel_count_downwards(insert_child, use_child.out_c)
          if not fixed:
            assert False, "Could not make channel counts of insert_child agree with required child"
          
      else: # Op is BinopIJK - keep the output channels of the chosen subtree
        #subtree = pick_subtree(tree, input_set, root_id=6)
        subtree = pick_subtree(tree, input_set)
      
      if not use_right:
        use_right = random.randint(0,1)
      if use_right:
        new_node = OpClass(cur_child, subtree)
      else:
        new_node = OpClass(subtree, cur_child)

      if use_child:
        # add a second parent if we're given a child to use
        subtree.parent = (subtree.parent, new_node)
      else:
        subtree.parent = new_node

    elif issubclass(OpClass, Unop):
      params = list(signature(OpClass).parameters.items())
      if len(params) == 2:
        out_c = random.sample(CHANNELS, 1)[0]
        new_node = OpClass(cur_child, out_c)
      elif len(params) == 1:
        new_node = OpClass(cur_child)
      else:
        assert False, "Invalid number of parameters for Unary op"

    cur_child.parent = new_node

    new_nodes = [new_node] + new_nodes
    cur_child = new_node
    cur_child.compute_input_output_channels()

  cur_child.parent = insert_parent

  if type(insert_parent) is tuple:
    for p in insert_parent:
      logging.debug("insert parent {} has {} children".format(p.name, p.num_children))
      link_insertion_parent(p, insert_child, cur_child)
  else:
    link_insertion_parent(insert_parent, insert_child, cur_child)
    logging.debug("insert parent {} has {} children".format(insert_parent.name, insert_parent.num_children))

  # make sure output channels of inserted nodes agree with parent
  # go back up to 
  cur_child.compute_input_output_channels()
  fixed = fix_channel_count_upwards(cur_child, cur_child.out_c)
  if not fixed:
    logging.debug("unable to make inserted nodes channel counts agree with tree")
    logging.debug("cur child {}".format(cur_child.dump()))
    logging.debug("tree {}".format(tree.dump()))
    assert False, "Could not make channel counts agree with inserted ops"
  else:
    logging.debug("fixed channel counts upwards")
    logging.debug("cur child {}".format(cur_child.dump()))
    logging.debug("tree {}".format(tree.dump()))
  tree.compute_input_output_channels()
  check_channel_count(tree)
  tree.add_dependees()

  return new_nodes

"""
selectively rejects an insertion location for given ops to inssert
"""
def accept_insertion_loc(child, parent, insert_ops):
  if insert_ops[-1] is Downsample:
    if not isinstance(child, Input):
      return False
  if insert_ops[-1] is LogSub:
    insertion_loc_options = find_closest_ancestor(child, set((Binop, Softmax)))
    if len(insertion_loc_options) < 3:
      return False
  return True

"""
rejects insertion ops if only Relu or Softmax is being inserted
These ops only make sense when used in combination with other ops
"""
def accept_insertion_op(insert_ops):
  return not (len(insert_ops) == 1 and (insert_ops[0] is Relu or insert_ops[0] is Softmax)) 

def accept_insertion_choice(child, parent, insert_ops):
  if not accept_insertion_loc(child, parent, insert_ops):
    logging.debug("rejecting inserting {} with child {}".format(insert_ops, child.dump()))
    return False
  if not accept_insertion_op(insert_ops):
    logging.debug("rejecting inserting {}".format(insert_ops))
    return False
  return True


def format_for_insert(insert_op):
  if type(insert_op) is tuple:
    assert len(insert_op) == 3, "Formatted insert ops must be a tuple of len 3" 
    return insert_op
  return (insert_op, None, None)


"""
rejects stupid trees
"""
def accept_tree(tree):
  if len(tree.preorder()) > MAX_SIZE:
    return False

  tree_type = type(tree) 

  if tree_type is SumR: 
    # don't sum reduce over one channel
    if tree.child.out_c == 1:
      logging.debug("rejecting SumR over one channel")
      return False
    # don't use sum reduce if model immediately expands channel count again afterwards
    parent = tree.parent
    banned_parents = set((Conv1x1, Conv1D, Conv2D, Softmax))
    while parent:
      # make sure the first parent seen is not a conv or a softmax - ignoring relus as parents
      if type(parent) in banned_parents:
        logging.debug("Rejecting SumR with parent Conv / Softmax")
        return False
      elif (not type(parent) is Relu) and (not type(parent) in banned_parents):
        break
      parent = parent.parent

  # don't allow subtraction or addition of same trees
  if tree_type is Sub or tree_type is Add:
    if tree.lchild.is_same_as(tree.rchild):
      logging.debug("Rejecting sub or add same tree")
      return False
    # don't allow addition or subtraction of linear subtrees
    if is_linear(tree):
      logging.debug("Rejecting sub or add linear trees")
      return False
  
  # don't allow LogSub of same trees
  if tree_type is LogSub:
    if tree.lchild.is_same_as(tree.rchild):
      logging.debug("rejecting LogSub same tree")
      return False

  # don't allow stacking of same trees
  # don't allow stack to stack trees that could have been one tree with channel count changes
  if tree_type is Stack:
    if tree.lchild.is_same_as(tree.rchild):
      logging.debug("rejecting stacking same children")
      return False
    if tree.lchild.is_same_mod_channels(tree.rchild):
      logging.debug("rejecting stacking structurally same children")
      return False

  # don't allow nested upsample / downsample and don't allow two softmaxes in same branch
  if tree_type is Upsample or tree_type is Softmax:
    ancestors_from_binop_to_node = find_closest_ancestor(tree, set((Binop,)))
    instances = sum([1 for n in ancestors_from_binop_to_node if type(n) is tree_type])
    if instances > 1:
      logging.debug("rejecting nested up/down or softmax")
      return False

  # don't allow adjacent LogSub / AddExp or adjacent Downsample / Upsample
  if tree_type is LogSub:
    if type(tree.parent) is AddExp:
      logging.debug("rejecting adjacent LogSub / AddExp")
      return False
  if tree_type is Downsample:
    if type(tree.parent) is Upsample:
      logging.debug("rejecting adjacent Down / Up")
      return False
    # downsample must be parent of input
    if not type(tree.child) is Input:
      logging.debug("rejecting downsample with non input child")
      return False

  # Mul must have either Softmax or Relu as one of its children
  # and do not allow two Mul in a row
  if tree_type is Mul:
    if type(tree.rchild) is tree_type or type(tree.lchild) is tree_type:
      logging.debug("rejecting two consecutive Mul")
      return False
    relu_or_softmax = set((Relu, Softmax))
    childtypes = set((type(tree.lchild), type(tree.rchild)))
    if len(relu_or_softmax.intersection(childtypes)) == 0:
      logging.debug("rejecting Mul without ReLU or Softmax child")
      return False

  if tree.num_children == 0:
    return True
  elif tree.num_children == 2:
    return accept_tree(tree.lchild) and accept_tree(tree.rchild)
  elif tree.num_children == 1:
    return accept_tree(tree.child)


"""
Tries to mutate tree by inserting random op(s) at a random location
If op is a binary op, picks a subtree to add as the other child
"""
def insert_mutation(tree, input_set, insert_above_node_id=None, insert_op=None):
  logging.debug("------------")
  preorder_nodes = tree.preorder()

  rejections = 0
  while True:
    if insert_above_node_id is None or rejections > 0:
      # insert above the selected node
      insert_above_node_id = random.randint(1, len(preorder_nodes)-1)
    insert_child = preorder_nodes[insert_above_node_id]
    insert_parent = insert_child.parent

    # pick op(s) to insert
    if insert_op is None:
      insert_types = get_insert_types(insert_parent, insert_child)
      while True:
        new_ops = []
        for n in range(len(insert_types)):
          OpClass = random.sample(insert_types[n], 1)[0]
          new_ops += [OpClass]
        if sum(map(lambda x : x in sandwich_ops, new_ops)) <= 1: 
          # allow at most one sandwich op to be inserted
          break
    else:
      new_ops = [insert_op] 
  
    if accept_insertion_choice(insert_child, insert_parent, new_ops):
      break
     
    rejections += 1

  new_ops = [format_for_insert(o) for o in new_ops]
  new_nodes = insert(tree, new_ops, insert_child, input_set)
  # if we inserted a sandwich op, we must insert its partner node as well
  # if we inserted Mul, and neither child is Softmax or Relu, add Softmax or Relu as a child
  for (OpClass,_,_), new_node in zip(new_ops, new_nodes):
    if OpClass is Mul:
      relu_or_softmax = set((Relu, Softmax))
      childtypes = set((type(new_node.lchild), type(new_node.rchild)))
      if len(relu_or_softmax.intersection(childtypes)) == 0:
        use_relu = random.randint(0,1)
        use_left = random.randint(0,1)
        if use_relu:
          insert_ops = [(Relu, None, None)]
        else:
          insert_ops = [(Softmax, None, None)]
        if use_left:
          insert_child = new_node.lchild
        else:
          insert_child = new_node.rchild
        insert_nodes = insert(tree, insert_ops, insert_child, input_set)
        logging.debug("tree after inserting softmax / relu {}".format(tree.dump()))

    if OpClass in sandwich_ops:

      OpPartner = sandwich_pairs[OpClass]  
      # decide where to insert partner node
      if OpClass is LogSub:
        # don't insert AddExp as parent of a Softmax or a binary op
        insertion_loc_options = find_closest_ancestor(new_node, set((Binop,Softmax))) 
      else:
        # don't insert sandwich op as parent of a binary op
        insertion_loc_options = find_closest_ancestor(new_node, set((Binop,)))

      # try not to insert partner right above node - makes sandwich pointless
      # use poisson to favor far apart pairs (i.e. smaller node id)
      insert_above_node_id = np.random.poisson((len(insertion_loc_options)-1)//3)
      insert_above_node_id = max(1, min(len(insertion_loc_options)-1, insert_above_node_id))
      insert_child = insertion_loc_options[insert_above_node_id]
      if OpClass is LogSub:
        # NOTE: we are passing a REFERENCE to LogSub's child -> creating a DAG
        insert_ops = [(OpPartner, new_node.rchild, True)] # LogSub subtracts the right child so must add back
      else:
        insert_ops = [(OpPartner, None, None)]

      insert_nodes = insert(tree, insert_ops, insert_child, input_set)
      partner_node = insert_nodes[0]

      if issubclass(OpPartner, NonLinear):
        if isinstance(partner_node.parent, NonLinear):
          insert_ops = [(random.sample(linear_ops, 1)[0], None, None)] 
          insert_nodes = insert(tree, insert_ops, partner_node, input_set)
        elif isinstance(insert_child, NonLinear):
          insert_ops = [(random.sample(linear_ops, 1)[0], None, None)] 
          insert_nodes = insert(tree, insert_ops, insert_child, input_set)

      # assign pointers to partner nodes
      new_node.partner = partner_node
      partner_node.partner = new_node
  
  logging.debug("------------")
  return tree

"""
checks whether subtree obeys the restricted input set and can be modified 
to agree with given target output channels.
Returns a copy of the subtree with channels appropriately modified if necessary
"""
def allow_subtree(root, input_set, target_out_c=None):
  size = root.compute_size(set(), count_input_exprs=True)
  # reject small trees
  if size < 2 or size > 11: 
      assert False, "Subtree of size {} is too small or too large".format(size)

  # reject trees that require inputs out of sanctioned input_set
  subtree_inputs = root.get_inputs()
  if not subtree_inputs.issubset(input_set):
    assert False, "rejecting subtree with invalid inputs"

  #root_copy = copy.deepcopy(root)
  root_copy = copy_subtree(root)
  # reject trees that can't be made to aggree with desired output channels
  if target_out_c:
    in_c, out_c = root_copy.compute_input_output_channels()
    if out_c != target_out_c: 
      fixed = fix_channel_count_downwards(root_copy, target_out_c)
      if fixed:
        in_c, out_c = root_copy.compute_input_output_channels()
      else:
        assert False, "rejecting subtree: cannot make output channels {} match {}".format(out_c, target_out_c)
  return root_copy
  

"""
selects a subtree from the given tree at root.
Returns a copy of that subtree without any pointers to its parent tree
"""
def pick_subtree(root, input_set, target_out_c=None, root_id=None):
  preorder = root.preorder()
  failures = 0
  while True:
    if root_id and failures == 0:
      subtree_id = root_id
    else:
      subtree_id = random.randint(1, len(preorder)-1)
    subtree = preorder[subtree_id]
    in_c, out_c = subtree.compute_input_output_channels()
    try:
      subtree_copy = allow_subtree(subtree, input_set, target_out_c)
    except AssertionError:
      failures += 1
      logging.debug("selected subtree is invalid")
    else:
      logging.debug("---selected subtree---")
      logging.debug(subtree_copy.dump())

      subtree_copy.parent = None
      return subtree_copy



