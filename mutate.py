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
from enum import Enum


CHANNELS = set((1,4,8,12))
CHANNELS = set((8,))

class MutationType(Enum):
  DELETION = 1
  INSERTION = 2


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
  def __init__(self, args, debug_logger, mysql_logger):
    
    self.args = args
    self.debug_logger = debug_logger
    self.mysql_logger = mysql_logger
    
    self.seen_models = {}
    self.seen_structures = set()

    self.current_mutation_info = {}
    self.failed_mutation_info = []

  def add_seen_model(self, tree, model_id):
    self.seen_models[tree] = {"model_ids":[int(model_id)], "hash": hash(tree), "ast_str": tree.dump()}

  def give_up(self, structural_rejections, prune_rejections, seen_rejections, failures):
    total_tries = structural_rejections + prune_rejections + seen_rejections + failures 
    if total_tries > self.args.mutation_failure_threshold:
      return True      
    else:
      return False      

  def pick_mutation_type(self, tree):
    tree_size = tree.compute_size(set(), count_all_inputs=True)
    if tree_size > self.args.max_nodes:
      mutation_type = MutationType.DELETION
    elif tree_size <= 3:
      mutation_type = MutationType.INSERTION
    else:
      mutation_type = random.choice(list(MutationType))
    return mutation_type

  def mutate(self, model_id, tree, input_set):
    failures = 0
    prune_rejections = 0
    structural_rejections = 0
    seen_rejections = 0
    
    self.failed_mutation_info = []

    while True: # loop to keep trying over assertion errors
      # if self.give_up(structural_rejections, prune_rejections, seen_rejections, failures):
      #   stats = MutationStats(failures, prune_rejections, structural_rejections, seen_rejections)
      #   return None, None, stats
      try:
        while True: # loop to keep trying over tree pruning 
          mutation_type = self.pick_mutation_type(tree)
          try:
            tree_copy = copy_subtree(tree)
          except AssertionError:
            self.debug_logger.debug(f"failed to copy model {model_id}")
            failures += 1
            continue

          self.current_mutation_info = {}
          self.current_mutation_info["mutation_type"] = mutation_type.name

          if mutation_type is MutationType.INSERTION:  
            new_tree = self.insert_mutation(tree_copy, input_set)
          else:
            new_tree = self.delete_mutation(tree_copy)
    
          if accept_tree(new_tree):
            break

          prune_rejections += 1
          if self.give_up(structural_rejections, prune_rejections, seen_rejections, failures):
            stats = MutationStats(failures, prune_rejections, structural_rejections, seen_rejections)
            self.failed_mutation_info.append(self.current_mutation_info.copy())
            return None, None, stats

      except (AssertionError, AttributeError) as e:
        if mutation_type is MutationType.INSERTION:
          self.debug_logger.debug(f'insertion mutation failed on model {model_id}')
        else:
          self.debug_logger.debug(f'deletion mutation failed on model {model_id}')
        self.failed_mutation_info.append(self.current_mutation_info.copy())
        failures += 1
        continue
      else: # successfully mutated tree
        # try to shrink channel counts - nodes like Stack can cause channel counts to blow up
        try:
          saved_copy = copy_subtree(new_tree)
        except AssertionError:
          self.debug_logger.debug(f"failed to copy model {model_id}")
          self.failed_mutation_info.append(self.current_mutation_info.copy())
          failures += 1
          continue
        try:
          shrink_channels(new_tree, self.args.default_channels, out_c=1)
        except AssertionError:
          self.debug_logger.debug("unable to shrink channels")
          new_tree = saved_copy

        new_tree.compute_input_output_channels()

        # TODO: THESE SHOULD NEVER FIRE BUT THEY DO... FIGURE OUT BUGS!!!
        # FOR NOW WE JUST CATCH THE ASSERTION ERRORS
        try:
          check_channel_count(new_tree) # these should not fire...
        except AssertionError:
          self.debug_logger.debug(f"channel count check failed on model {model_id}")
          self.failed_mutation_info.append(self.current_mutation_info.copy())
          failures += 1
          continue
        try:
          check_linear_types(new_tree)
        except AssertionError:
          self.debug_logger.debug(f"check linearity failed on model {model_id}")
          self.failed_mutation_info.append(self.current_mutation_info.copy())
          failures += 1
          continue

        # reject trees we've seen already 
        if (not new_tree in self.seen_models):
          # reject with some chance trees with previously seen structure
          h = structural_hash(new_tree)
          if h in self.seen_structures:
            if random.random() < self.args.structural_sim_reject:
              structural_rejections += 1
              continue
          else: # successfully mutated tree!
            self.seen_structures.add(h)
            self.seen_models[new_tree] = {"model_ids":[int(model_id)], "hash": hash(new_tree), "ast_str": new_tree.dump()}
            stats = MutationStats(failures, prune_rejections, structural_rejections, seen_rejections)
            return new_tree, h, stats

        if new_tree in self.seen_models: # we've seen and evaluated this exact tree before
          self.seen_models[new_tree]["model_ids"].append(int(model_id))
          seen_rejections += 1
          continue


"""
returns whether parent and child linearity type sequence is allowed
"""
def legal_parent_child_linearity(parent, child):
  if type(parent) is tuple:
    if not (type(parent[0]) is type(parent[1])):
      logging.debug("Node with multiple parents cannot have parents of different types")
      assert False, "If a node has multiple parents they must have same type"
    parent_type = type(parent[0])
  else:
    parent_type = type(parent)
  child_type = type(child)
  return parent_type in border_ops \
      or child_type in border_ops \
      or (parent_type in nl_and_sp and child_type in l_and_sp) \
      or (parent_type in l_and_sp and child_type in nl_and_sp)
  

"""
adds partner node to give node's set of partner nodes
"""
def add_partner(node, node_partner):
  if hasattr(node, "partner_set"):
    node.partner_set.add((node_partner, id(node_partner)))
  else:
    node.partner_set = set([(node_partner, id(node_partner))])

"""
given set of nodes, return all of their partner nodes
"""  
def get_partner_nodes(nodes_and_ids):
  partner_nodes = set()
  for n, nid in nodes_and_ids:
    if hasattr(n, "partner_set"):
      new_set = set()
      for item in n.partner_set: 
        # nodes may have been modified since insertion into partner set
        # so we need to reproduce the set by rehashing the nodes
        new_set.add(item)
      partner_nodes = partner_nodes.union(new_set)
  return partner_nodes

"""
given a set of nodes that are partners of (nodes, ids) that have just been deleted, 
remove their connections to their deleted partners
"""
def remove_deleted_partner_connections(partners_of_deleted, deleted):
  for n, nid in partners_of_deleted:
    new_set = set()
    for item in n.partner_set: 
      # nodes may have been modified since insertion into partner set
      # so we need to reproduce the set by rehashing the nodes
      new_set.add(item)
    n.partner_set = new_set - deleted
    if len(n.partner_set) == 0:
      delattr(n, "partner_set")

"""
fixes up parent child relationships to remove given node
"""
def remove_node(parent, deletion_node, new_child):
  if parent.num_children == 3:
    if deletion_node is parent.child1:
      parent.child1 = new_child
    elif deletion_node is parent.child2:
      parent.child2 = new_child
    else:
      parent.child3 = new_child 
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
  if deletion_parent.num_children == 3:
    if deletion_child is deletion_parent.child1:
      out_c = deletion_parent.in_c[0]
    elif deletion_child is deletion_parent.child2:
      out_c = deletion_parent.in_c[1]
    else:
      out_c = deletion_parent.in_c[2]
  if deletion_parent.num_children == 2:
    if deletion_child is deletion_parent.lchild:
      out_c = deletion_parent.in_c[0]
    else:
      out_c = deletion_parent.in_c[1]
  else:
    out_c = deletion_parent.in_c

  # try to fix channels from child down
  # make a copy in case fixing fails and we need to try fixing upwards
  in_out_channels = deletion_child.get_input_output_channels()
  fixed = fix_channel_count_downwards(deletion_child, out_c)

  if fixed:
    deletion_child.compute_input_output_channels()
    check_channel_count(tree)
  else:
    # reset input output channels
    deletion_child.set_input_output_channels(in_out_channels)
    # try to fix channels from parent up
    deletion_child.compute_input_output_channels()
    fixed = fix_channel_count_upwards(deletion_child, deletion_child.out_c)
    if fixed:
      tree.compute_input_output_channels()
      check_channel_count(tree)
    else:
      logging.debug("Could not make channel counts agree after deleting ops")
      assert False, "Could not make channel counts agree after deleting ops"


"""
deletes nodes from tree starting from the given node up the tree until
linearity rules are obeyed.
reassigns deleted subtrees to one if any of its dependees
returns the child of the deleted nodes
"""
@extclass(Mutator)
def delete_nodes(self, tree, node, already_deleted=None):
  if already_deleted is None:
    already_deleted = {}

  parent = node.parent
  cur_node = node
  deleted_nodes = set()

  while True:
    deleted_nodes.add((cur_node, id(cur_node)))
    if isinstance(cur_node, Binop):
      # if op is LogSub or AddExp always keep left child
      if type(cur_node) is LogSub or type(cur_node) is AddExp:
        keep_left = True
      else:
        # decide if you're going to keep left or right child
        keep_left = random.randint(0,1)
      if keep_left: 
        child = cur_node.lchild
        deleted_nodes = deleted_nodes.union(set([(dn, id(dn)) for dn in cur_node.rchild.preorder()]))
      else:
        child = cur_node.rchild
        deleted_nodes = deleted_nodes.union(set([(dn, id(dn)) for dn in cur_node.lchild.preorder()]))

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

  # delete partner if we deleted a sandwich node
  partners_of_deleted = get_partner_nodes(deleted_nodes)

  partners_of_deleted = partners_of_deleted - deleted_nodes # don't need to delete partners already deleted

  # add newly deleted nodes to already_deleted
  for dn in deleted_nodes:
    already_deleted[id(dn)] = dn

  filtered_partners_of_deleted = set()
  for p in partners_of_deleted:
    if not id(p) in already_deleted:
      filtered_partners_of_deleted.add(p)

  # must remove partner connections that refer back to deleted_nodes - else infinite recursion deletion
  remove_deleted_partner_connections(filtered_partners_of_deleted, deleted_nodes)
  for n, nid in filtered_partners_of_deleted:
    tree = self.delete_nodes(tree, n, already_deleted)
  
  fix_channels_after_deletion(tree, parent, child)

  return tree


"""
picks a node to delete from a tree - ok as long as not a border node
rejects nodes for deletion with increasing probability based on how 
many nodes would be removed at most from its deletion
for example, if deleting a binary node, we could remove as many nodes as its
largest subtree
"""
@extclass(Mutator)
def select_node_to_delete(self, tree):
  preorder_nodes = tree.preorder()
  n = len(preorder_nodes)
  tree.compute_size(set(), count_all_inputs=True)
  # rejections = 0
 
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
    # rejections += 1
    # if rejections > self.args.delete_failure_threshold:
    #   return None
  return node


"""
picks a randomm op in tree to delete
"""
@extclass(Mutator)
def delete_mutation(self, tree, node=None):
  # delete the selected node
  if node is None:
    node = self.select_node_to_delete(tree)
  if node is None:
    return None

  self.current_mutation_info["delete_id"] = tree.get_preorder_id(node)
  tree = self.delete_nodes(tree, node)
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
      logging.debug("Linear parent cannot have Linear child")
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
      logging.debug("NonLinear parent cannot have NonLinear child")
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
  if insert_parent.num_children == 3:
    if insert_child is insert_parent.child1:
      insert_parent.child1 = node
    elif insert_child is insert_parent.child2:
      insert_parent.child2 = node
    else:
      insert_parent.child3 = node
  if insert_parent.num_children == 2:
    if insert_child is insert_parent.rchild:
      insert_parent.rchild = node
    else:
      insert_parent.lchild = node
  elif insert_parent.num_children == 1:
    insert_parent.child = node
  else:
    logging.debug("Should not be inserting after an input node")
    assert False, "Should not be inserting after an input node"

@extclass(Mutator)
def insert_binop(self, tree, input_set, insert_op, insert_child):
  OpClass, use_child, use_right = insert_op
  if not hasattr(tree, "size"):
    tree.compute_size(set(), count_all_inputs=True)

  #### pick a subtree ####
  if issubclass(OpClass, BinopIII):
    # make subtree's output channels match other child's output channels
    if use_child is None:
      if tree.size < self.args.min_subtree_size + 2:
        logging.debug(f"Impossible to find subtree of size {self.args.min_subtree_size} or greater from tree of size {tree.size}")
        assert False, f"Impossible to find subtree of size {self.args.min_subtree_size} or greater from tree of size {tree.size}"
      subtree = self.pick_subtree(tree, input_set, target_out_c=insert_child.out_c, resolution=spatial_resolution(insert_child))
      subtree.compute_input_output_channels()         
    else:
      subtree = use_child
      # make channel count of insert_child agree with required use_child
      fixed = fix_channel_count_downwards(insert_child, use_child.out_c)
      if not fixed:
        logging.debug(f"Could not make channel counts of insert_child agree with required child")
        assert False, f"Could not make channel counts of insert_child agree with required child"
  else: # Op is BinopIJK - keep the output channels of the chosen subtree
    # subtree = self.pick_subtree(tree, input_set, root_id=6)
    if tree.size < self.args.min_subtree_size + 2:
      logging.debug(f"Impossible to find subtree of size {self.args.min_subtree_size} or greater from tree of size {tree.size}")
      assert False, f"Impossible to find subtree of size {self.args.min_subtree_size} or greater from tree of size {tree.size}"
    subtree = self.pick_subtree(tree, input_set, resolution=spatial_resolution(insert_child))
  

  # check spatial resolutions of subtree and insert_child match 
  subtree_res = spatial_resolution(subtree)
  insert_child_res = spatial_resolution(insert_child)

  if subtree_res != insert_child_res:
    logging.debug(f"resolution of chosen subtree for binary op does not match insert_child resolution")
    assert False, f"resolution of chosen subtree for binary op does not match insert_child resolution"
  elif subtree_res == Resolution.DOWNSAMPLED:
    # add new subtree partner nodes to the sets of partner nodes in the insertion parent(s) 
    upsample_parents = find_closest_parents(insert_child, set((Upsample,)))
    downsample_children = find_closest_children(subtree, set((Downsample,)))
 
    for upsample_parent, up_id in upsample_parents:
      upsample_parent.partner_set = upsample_parent.partner_set.union(downsample_children)
   
    for downsample_child, dc_id in downsample_children:
      downsample_child.partner_set = downsample_child.partner_set.union(upsample_parents)

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
  if len(params) == 3: # not sure what's going on here with number of params...
    out_c = insert_child.out_c # self.defualt_channels
    new_node = OpClass(insert_child, out_c)
  elif len(params) == 2:
    new_node = OpClass(insert_child)
  else:
    logging.debug("Invalid number of parameters for Unary op")
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
@extclass(Mutator)
def choose_partner_op_loc(self, op_node, OpClass, partner_op_class):
  # decide where to insert partner node
  if OpClass is LogSub:
    # don't insert AddExp as parent of a Softmax or a binary op
    insertion_child_options = find_closest_ancestor(op_node, set((Binop,Softmax)))[1:] 
    resolution = spatial_resolution(op_node.rchild)
  else:
    # don't insert sandwich op as parent of a binary op
    insertion_child_options = find_closest_ancestor(op_node, set((Binop,)))[1:]
    resolution = None

  tries = 0
  while True:
    # use poisson with small lambda to favor locations farther away (i.e. smaller index)
    insert_above_node_loc = np.random.poisson(len(insertion_child_options)//3)
    insert_above_node_loc = min(len(insertion_child_options)-1, insert_above_node_loc)
    insert_child = insertion_child_options[insert_above_node_loc]
    insert_child_res = spatial_resolution(insert_child)
    if resolution is None or insert_child_res == resolution:
      break
    tries += 1
    if tries > self.args.select_insert_loc_tries:
      logging.debug(f"unable to find an insertion location for partner op of {op_node} with resolution {resolution}")
      assert False, f"unable to find an insertion location for partner op of {op_node} with resolution {resolution}"

  if OpClass is LogSub:
    # NOTE: we are passing a REFERENCE to LogSub's child -> creating a DAG
    insert_ops = [(partner_op_class, op_node.rchild, True)] # LogSub subtracts the right child so must add back
  else:
    insert_ops = [(partner_op_class, None, None)]

  return insert_ops, insert_child

"""
inserts the partner op of the given op_node with class op_class
"""
@extclass(Mutator)
def insert_partner_op(self, tree, input_set, op_class, op_node):
  op_partner_class = sandwich_pairs[op_class]  
  try:
    insert_op, insert_child = self.choose_partner_op_loc(op_node, op_class, op_partner_class)
  except AssertionError:
    logging.debug(f"failed to find insertion location for partner of {op_node}")
    assert False, f"failed to find insertion location for partner of {op_node}"
  
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
  add_partner(op_node, partner_node)
  add_partner(partner_node, op_node)

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
      assert len(params) == 3, f"Invalid number of parameters {len(params)} for Binary op"
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

  self.current_mutation_info["insert_ops"] = ",".join([new_op[0].__name__ for new_op in new_ops])
  self.current_mutation_info["insert_child_id"] = insert_above_node_id

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


def has_downsample(tree):
  if isinstance(tree, Downsample):
    return True
  elif tree.num_children == 3:
    c1_has = has_downsample(tree.child1)
    c2_has = has_downsample(tree.child2)
    c3_has = has_downsample(tree.child3)
    return c1_has or c2_has or c3_has
  elif tree.num_children == 2:
    l_has = has_downsample(tree.lchild)
    r_has = has_downsample(tree.rchild)
    return l_has or r_has
  elif tree.num_children == 1:
    return has_downsample(tree.child)
  else:
    return False

def has_upsample(tree):
  if isinstance(tree, Upsample):
    return True
  elif tree.num_children == 3:
    c1_has = has_upsample(tree.child1)
    c2_has = has_upsample(tree.child2)
    c3_has = has_upsample(tree.child3)
    return c1_has or c2_has or c3_has
  elif tree.num_children == 2:
    l_has = has_upsample(tree.lchild)
    r_has = has_upsample(tree.rchild)
    return l_has or r_has
  elif tree.num_children == 1:
    return has_upsample(tree.child)
  else:
    return False

"""
detects if subtree splits up any sandwich ops

Disallow subtrees from splitting LogSub and AddExp
but splitting Down/Upsample is ok
"""
def splits_sandwich_ops(tree):
  partner_pool = splits_sandwich_ops_helper(tree)
  partner_ids = [t[1] for t in partner_pool]
  # sandwich ops are split if ids of all nodes in partner pool 
  # are not present among the ids in the tree
  tree_ids = set([id(n) for n in tree.preorder()])
  if len(partner_pool - tree_ids) != 0:
    return True
  return False 
  
def splits_sandwich_ops_helper(tree, partner_pool=None):
  if partner_pool is None:
    partner_pool = set()
  if hasattr(tree, "partner_set") and not \
  (isinstance(tree, Downsample) or isinstance(tree, Upsample)):
    partner_pool = partner_pool.union(tree.partner_set)

  if tree.num_children == 3:
    child1_pool = splits_sandwich_ops_helper(tree.child1, partner_pool)
    child2_pool = splits_sandwich_ops_helper(tree.child2, child1_pool)
    partner_pool = splits_sandwich_ops_helper(tree.child3, child2_pool) 
  if tree.num_children == 2:
    lchild_pool = splits_sandwich_ops_helper(tree.lchild, partner_pool)
    partner_pool = splits_sandwich_ops_helper(tree.rchild, lchild_pool)
  if tree.num_children == 1:
    partner_pool = splits_sandwich_ops_helper(tree.child, partner_pool)
  return partner_pool


"""
checks whether subtree obeys the restricted input set and can be modified 
to agree with given target output channels.
if resolution is given, checks spatial resolution of subtree matches
Returns a copy of the subtree with channels appropriately modified if necessary
"""
@extclass(Mutator)
def allow_subtree(self, root, input_set, target_out_c=None, resolution=None):
  subtree_resolution = spatial_resolution(root)
  if subtree_resolution != resolution:
    logging.debug(f"subtree with resolution {subtree_resolution} does not match required resolution {resolution}")
    assert False, f"subtree with resolution {subtree_resolution} does not match required resolution {resolution}"
  size = root.compute_size(set(), count_input_exprs=True)
  # reject small trees or large trees
  if size < self.args.min_subtree_size or size > self.args.max_subtree_size: 
      logging.debug(f"Subtree of size {size} is too small or too large")
      assert False, f"Subtree of size {size} is too small or too large"

  # reject trees that require inputs out of sanctioned input_set
  subtree_inputs = root.get_inputs()
  if not subtree_inputs.issubset(input_set):
    logging.debug("rejecting subtree with invalid inputs")
    assert False, "rejecting subtree with invalid inputs"

  # reject subtrees that split up sandwich nodes LogSub and AddExp
  # NOTE : ALLOWING SUBTREES THAT SPLIT DOWN/UPSAMPLE IS CONDITIONED ON SPATIAL RESOLUTIONS MATCHING
  splits = splits_sandwich_ops(root)
  assert (not splits), f"rejecting subtree: splits sandwich_ops"
  
  root_copy = copy_subtree(root)
  
  # reject trees that can't be made to aggree with desired output channels
  if target_out_c:
    in_c, out_c = root_copy.compute_input_output_channels()
    if out_c != target_out_c: 
      fixed = fix_channel_count_downwards(root_copy, target_out_c)
      if fixed:
        root_copy.compute_input_output_channels()
      else:
        logging.debug(f"rejecting subtree: cannot make output channels {out_c} match {target_out_c}")
        assert False, f"rejecting subtree: cannot make output channels {out_c} match {target_out_c}"
  return root_copy
  

"""
selects a subtree from the given tree at root.
Returns a copy of that subtree without any pointers to its parent tree
"""
@extclass(Mutator)
def pick_subtree(self, root, input_set, target_out_c=None, resolution=None, root_id=None):
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
      subtree_copy = self.allow_subtree(subtree, input_set, target_out_c, resolution)
    except AssertionError:
      failures += 1
      logging.debug("selected subtree is invalid")
      if failures > self.args.subtree_selection_tries:
        logging.debug(f"TOO MANY TRIES TO FIND SUBTREE with resolution {resolution} from:")
        logging.debug(root.dump())
        assert False, f"Too many tries to find subtree with resolution {resolution}"
    else:
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
  elif tree.num_children == 3:
    return accept_tree(tree.child1) and accept_tree(tree.child2) and accept_tree(tree.child3)

        
