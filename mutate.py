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
from util import extclass


LOG_FILENAME='log.txt'
logger = logging.getLogger("DebugLogger")
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

CHANNELS = set((1,4,8,12))
CHANNELS = set((8,))

class MutationStats():
  def __init__(self, failures, prune_rejections, structural_rejections, seen_rejections):
    self.failures = failures
    self.prune_rejections = prune_rejections
    self.structural_rejections = structural_rejections
    self.seen_rejections = seen_rejections

"""
mutates the given tree 
50/50 percent chance of doing an insertion or deletion mutation
unless tree size exceeds MAX_NODES. If tree size exceeds MAX_NODES 
only performs deletion mutation
"""
class Mutator():
  def __init__(self, args, debug_logger):
    
    self.args = args
    self.debug_logger = debug_logger

    self.seen_models = {}
    self.seen_structures = set()

  def mutate(self, model_id, tree, input_set):
    failures = 0
    prune_rejections = 0
    structural_rejections = 0
    seen_rejections = 0

    tree_size = tree.compute_size(set(), count_all_inputs=True)
    
    while True:
      total_tries = structural_rejections + prune_rejections + seen_rejections + failures 
      if total_tries > self.args.mutation_failure_threshold:
        stats = MutationStats(failures, prune_rejections, structural_rejections, seen_rejections)
        print(f'killed mutation from {model_id}')
        return None, None, stats
      try:
        while True:
          if tree_size > self.args.max_nodes:
            do_insertion = False
          else:
            do_insertion = random.randint(0,1)
          tree_copy = copy.deepcopy(tree)

          if do_insertion:  
            new_tree = self.insert_mutation(tree_copy, input_set)
          else:
            new_tree = self.delete_mutation(tree_copy)
          if accept_tree(new_tree):
            break
          prune_rejections += 1
      except AssertionError:
        if do_insertion:
          self.debug_logger.debug(f'insertion mutation failed on model {model_id}')
        else:
          self.debug_logger.debug(f'deletion mutation failed on model {model_id}')
        failures += 1
        continue
      else: # successfully mutated tree
        # reject trees we've seen already 
        if not new_tree in self.seen_models:
          # reject with some chance trees with previously seen structure
          h = structural_hash(new_tree)
          if h in self.seen_structures:
            if random.random() < self.args.structural_sim_reject:
              structural_rejections += 1
              continue
          else: # successfully mutated tree!
            check_channel_count(new_tree) # these should not fire...
            check_linear_types(new_tree)
            self.seen_structures.add(h)
            self.seen_models[new_tree] = [int(model_id)]
            stats = MutationStats(failures, prune_rejections, structural_rejections, seen_rejections)
            return new_tree, h, stats

        else: # we've seen and evaluated this exact tree before
          self.seen_models[new_tree].append(int(model_id))
          seen_rejections += 1
          continue


"""
returns whether parent and child linearity type sequence is allowed
"""
def legal_parent_child_linearity(parent, child):
  if type(parent) is tuple:
    assert type(parent[0]) is type(parent[1]), "If a node has multiple parents they must have same type"
    parent_type = type(parent[0])
  else:
    parent_type = type(parent)
  child_type = type(child)
  return parent_type in border_ops \
      or child_type in border_ops \
      or (parent_type in nl_and_sp and child_type in l_and_sp) \
      or (parent_type in l_and_sp and child_type in nl_and_sp)
  

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
  else:
    # try to fix channels from parent up
    deletion_child.compute_input_output_channels()
    fixed = fix_channel_count_upwards(deletion_child, deletion_child.out_c)
    if fixed:
      tree.compute_input_output_channels()
      check_channel_count(tree)
    else:
      assert False, "Could not make channel counts agree after deleting ops"


"""
deletes nodes from tree starting from the given node up the tree until
linearity rules are obeyed.
reassigns deleted subtrees to one if any of its dependees
returns the child of the deleted nodes
"""
@extclass(Mutator)
def delete_nodes(self, tree, node):
  parent = node.parent
  cur_node = node
  deleted_nodes = set()

  while True:
    deleted_nodes.add(cur_node)
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

    if parent is None:
      break

    if type(parent) is tuple:
      for p in parent:
        remove_node(p, cur_node, child)
    else:
      remove_node(parent, cur_node, child)
     
    if legal_parent_child_linearity(parent, child):
      break

    # deletion caused illegal linearity, move up the tree to delete next node
    if type(parent) is tuple: 
      # parents are LogSub and AddExp - one of which will be deleted in next loop iteration
      # the other parent will be deleted by the code below that handles deleting partners
      cur_node = cur_node.parent[0]
    else:
      cur_node = cur_node.parent

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
rejects nodes for deletion with increasing probability based on how 
many nodes would be removed at most from its deletion
for example, if deleting a binary node, we could remove as many nodes as its
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
@extclass(Mutator)
def delete_mutation(self, tree):
  # delete the selected node
  node = select_node_to_delete(tree)
  deletion_child = self.delete_nodes(tree, node)
  tree.compute_input_output_channels()
  return tree 




"""

functions for supporting insertion 

"""


"""
Returns sets of valid op type(s) to insert given the types of the 
parent and child nodes: Linear, Special, or NonLinear.
May return one or two sets. If one set is returned, one node will be 
inserted with an op type belonging to the returned set.
If two sets are returned, two nodes will be inserted with their types
dictated by those two sets. 
"""
@extclass(Mutator)
def get_insert_types(self, parent, child):
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
connects insertion parent node to newly created child
insert_child: the node above which the insertion occured
insert_parent: the original parent of insert_child
node: the node inserted directly below insert_parent
"""
@extclass(Mutator)
def link_insertion_parent_to_new_child(self, insert_parent, insert_child, node):
  if insert_parent.num_children == 2:
    if insert_child is insert_parent.rchild:
      insert_parent.rchild = node
    else:
      insert_parent.lchild = node
  elif insert_parent.num_children == 1:
    insert_parent.child = node
  else:
    assert False, "Should not be inserting after an input node"

@extclass(Mutator)
def insert_binop(self, tree, input_set, insert_op, insert_child):
  OpClass, use_child, use_right = insert_op
  if issubclass(OpClass, BinopIII):
    # make subtree's output channels match other child's output channels
    if use_child is None:
      subtree = self.pick_subtree(tree, input_set, target_out_c=insert_child.out_c)
      subtree.compute_input_output_channels()         
    else:
      subtree = use_child
      # make channel count of insert_child agree with required use_child
      fixed = fix_channel_count_downwards(insert_child, use_child.out_c)
      if not fixed:
        assert False, f"Could not make channel counts of insert_child agree with required child"
      
  else: # Op is BinopIJK - keep the output channels of the chosen subtree
    #subtree = self.pick_subtree(tree, input_set, root_id=6)
    subtree = self.pick_subtree(tree, input_set)
  
  if not use_right:
    use_right = random.randint(0,1)

  if use_right:
    new_node = OpClass(insert_child, subtree)
  else:
    new_node = OpClass(subtree, insert_child)

  if use_child: # add a second parent if we're given a child to use
    subtree.parent = (subtree.parent, new_node)
  else:
    subtree.parent = new_node
  
  insert_child.parent = new_node
  return new_node

@extclass(Mutator)
def insert_unary_op(self, OpClass, insert_child):
  params = list(signature(OpClass).parameters.items())
  if len(params) == 2:
    # set output channels = input channels so that we can use skip connections
    out_c = insert_child.out_c # self.defualt_channels
    new_node = OpClass(insert_child, out_c)
  elif len(params) == 1:
    new_node = OpClass(insert_child)
  else:
    assert False, "Invalid number of parameters for Unary op"

  insert_child.parent = new_node
  return new_node

"""
selectively rejects an insertion location for given ops to insert
Downsamples must be inserted above input node
reject if LogSub would be inserted without enough space above to insert its partner
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
Chooses which activation function to insert under a Mul
"""
def choose_activation_under_mul(mul_node):
  use_relu = random.randint(0,1)
  use_left = random.randint(0,1)
  if use_relu:
    insert_ops = [(Relu, None, None)]
  else:
    insert_ops = [(Softmax, None, None)]
  if use_left:
    insert_child = mul_node.lchild
  else:
    insert_child = mul_node.rchild

  return insert_ops, insert_child

"""
inserts an activation function under a Mul node if necessary
"""
@extclass(Mutator)
def insert_activation_under_mul(self, tree, input_set, mul_node):
  relu_or_softmax = set((Relu, Softmax))
  childtypes = set((type(mul_node.lchild), type(mul_node.rchild)))
  if len(relu_or_softmax.intersection(childtypes)) == 0:
    insert_op, insert_child = choose_activation_under_mul(mul_node)
    insert_nodes = self.insert(tree, insert_op, insert_child, input_set)
  return 

"""
chooses were to insert sandwich op's partner
"""
def choose_partner_op_loc(op_node, OpClass, partner_op_class):
  # decide where to insert partner node
  if OpClass is LogSub:
    # don't insert AddExp as parent of a Softmax or a binary op
    insertion_child_options = find_closest_ancestor(op_node, set((Binop,Softmax)))[1:] 
  else:
    # don't insert sandwich op as parent of a binary op
    insertion_child_options = find_closest_ancestor(op_node, set((Binop,)))[1:]

  # use poisson with small lambda to favor locations farther away (i.e. smaller index)
  insert_above_node_loc = np.random.poisson(len(insertion_child_options)//3)
  insert_above_node_loc = min(len(insertion_child_options)-1, insert_above_node_loc)
  insert_child = insertion_child_options[insert_above_node_loc]

  if OpClass is LogSub:
    # NOTE: we are passing a REFERENCE to LogSub's child -> creating a DAG
    insert_ops = [(partner_op_class, op_node.rchild, True)] # LogSub subtracts the right child so must add back
  else:
    insert_ops = [(partner_op_class, None, None)]

  return insert_ops, insert_child

@extclass(Mutator)
def insert_partner_op(self, tree, input_set, op_class, op_node):
  op_partner_class = sandwich_pairs[op_class]  
  insert_op, insert_child = choose_partner_op_loc(op_node, op_class, op_partner_class)

  insert_nodes = self.insert(tree, insert_op, insert_child, input_set)
  partner_node = insert_nodes[0]

  # manually fix linear / nonlinearity by inserting an additional node
  if issubclass(op_partner_class, NonLinear):
    if isinstance(partner_node.parent, NonLinear):
      insert_ops = [(random.sample(linear_ops, 1)[0], None, None)] 
      insert_nodes = insert(tree, insert_ops, partner_node, input_set)
    elif isinstance(insert_child, NonLinear):
      insert_ops = [(random.sample(linear_ops, 1)[0], None, None)] 
      insert_nodes = insert(tree, insert_ops, insert_child, input_set)

  # assign pointers to partner nodes
  op_node.partner = partner_node
  partner_node.partner = op_node
  return 

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
@extclass(Mutator)
def insert(self, tree, insert_ops, insert_child, input_set):
  insert_parent = insert_child.parent
  cur_child = insert_child
  new_nodes = [] 

  for i, (OpClass, use_child, use_right) in enumerate(reversed(insert_ops)):
    cur_child.compute_input_output_channels()

    if issubclass(OpClass, Binop):
      params = list(signature(OpClass).parameters.items())
      assert len(params) == 2, "Invalid number of parameters for Binary op"
      new_node = self.insert_binop(tree, input_set, (OpClass, use_child, use_right), cur_child)

    elif issubclass(OpClass, Unop):
      new_node = self.insert_unary_op(OpClass, cur_child)
      
    new_nodes = [new_node] + new_nodes
    cur_child = new_node

  cur_child.parent = insert_parent

  if type(insert_parent) is tuple:
    for p in insert_parent:
      self.link_insertion_parent_to_new_child(p, insert_child, cur_child)
  else:
    self.link_insertion_parent_to_new_child(insert_parent, insert_child, cur_child)

  # make sure output channels of inserted nodes agree with parent
  cur_child.compute_input_output_channels()
  fixed = fix_channel_count_upwards(cur_child, cur_child.out_c)
  if not fixed:
    logging.debug("unable to make inserted nodes channel counts agree with tree")
    assert False, "Could not make channel counts agree with inserted ops"

  tree.compute_input_output_channels()
  check_channel_count(tree)

  return new_nodes


"""
Tries to mutate tree by inserting random op(s) at a random location
If op is a binary op, picks a subtree to add as the other child
"""
@extclass(Mutator)
def insert_mutation(self, tree, input_set, insert_above_node_id=None, insert_op=None):
  preorder_nodes = tree.preorder()

  while True:
    # insert above the selected node
    if not insert_above_node_id:
      insert_above_node_id = random.randint(1, len(preorder_nodes)-1)
    insert_child = preorder_nodes[insert_above_node_id]
    insert_parent = insert_child.parent

    # pick op(s) to insert
    insert_types = self.get_insert_types(insert_parent, insert_child)
   
    while True:
      if not insert_op:
        new_ops = []
        for n in range(len(insert_types)):
          OpClass = random.sample(insert_types[n], 1)[0]
          new_ops += [OpClass]
      else:
        new_ops = [insert_op]
      # allow at most one sandwich op to be inserted
      if sum(map(lambda x : x in sandwich_ops, new_ops)) <= 1: 
        break

    if accept_insertion_choice(insert_child, insert_parent, new_ops):
      break
     
  new_ops = [format_for_insert(o) for o in new_ops]
  new_nodes = self.insert(tree, new_ops, insert_child, input_set)

  # if we inserted a sandwich op, we must insert its partner node as well
  # if we inserted Mul, and neither child is Softmax or Relu, add Softmax or Relu as a child
  """
   NOTE: not ideal that this pruning rule is being muddled in as an insertion rule
  """
  for (OpClass,_,_), new_node in zip(new_ops, new_nodes):
    if OpClass is Mul:
      self.insert_activation_under_mul(tree, input_set, new_node)
    if OpClass in sandwich_ops:
      self.insert_partner_op(tree, input_set, OpClass, new_node)

  return tree

"""
checks whether subtree obeys the restricted input set and can be modified 
to agree with given target output channels.
Returns a copy of the subtree with channels appropriately modified if necessary
"""
@extclass(Mutator)
def allow_subtree(self, root, input_set, target_out_c=None):
  size = root.compute_size(set(), count_input_exprs=True)
  # reject small trees or large trees
  if size < self.args.min_subtree_size or size > self.args.max_subtree_size: 
      assert False, f"Subtree of size {size} is too small or too large"

  # reject trees that require inputs out of sanctioned input_set
  subtree_inputs = root.get_inputs()
  if not subtree_inputs.issubset(input_set):
    assert False, "rejecting subtree with invalid inputs"

  root_copy = copy_subtree(root)
  # reject trees that can't be made to aggree with desired output channels
  if target_out_c:
    in_c, out_c = root_copy.compute_input_output_channels()
    if out_c != target_out_c: 
      fixed = fix_channel_count_downwards(root_copy, target_out_c)
      if fixed:
        root_copy.compute_input_output_channels()
      else:
        assert False, f"rejecting subtree: cannot make output channels {out_c} match {target_out_c}"
  return root_copy
  

"""
selects a subtree from the given tree at root.
Returns a copy of that subtree without any pointers to its parent tree
"""
@extclass(Mutator)
def pick_subtree(self, root, input_set, target_out_c=None, root_id=None):
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
      subtree_copy = self.allow_subtree(subtree, input_set, target_out_c)
    except AssertionError:
      failures += 1
      logging.debug("selected subtree is invalid")
    else:
      logging.debug("---selected subtree---")
      logging.debug(subtree_copy.dump())

      subtree_copy.parent = None
      return subtree_copy




"""
Pruner
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

  # Conv1D must have output channels % 4 == 0
  if tree_type is Conv1D and tree.out_c % 4 != 0:
    return False

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

  # don't allow softmax over a single channel
  if tree_type is Softmax and tree.out_c == 1:
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


        
