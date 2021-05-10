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
import numpy as np
from demosaic_ast import *
from type_check import *
from tree import *
from restrictions import *
from util import extclass, get_factors, get_closest_factor
from enum import Enum
from footprint import compute_footprint
from search_mutation_type_pdfs import mutation_types_pdfs
from tree_manipulation import insertion_edge_updates, replace_child, replace_parent
import normalize
from resolution_coloring import change_subgraph_resolution, flip_resolution, delete_resolution_subgraph, swap_resolution_op
import traceback


channel_options = [8, 9, 12, 16, 18, 20, 24, 27, 28, 36]


class BinopOperandType(Enum):
  INPUTOP = 1 # take subtree from input op set
  RECURRENT = 2 # take subtree from mutating tree
  CROSSTREE = 3 # take subtree from partner tree


class MutationType(Enum):
  DELETION = 1
  INSERTION = 2
  DECOUPLE = 3
  CHANNEL_CHANGE = 4
  GROUP_CHANGE = 5
  INSERT_RESOLUTION_SUBGRAPH = 6
  DELETE_RESOLUTION_SUBGRAPH = 7
  SHIFT_RESOLUTION_SUGRAPH = 8
  SWAP_RESOLUTION_OP = 9
  GREEN_MODEL_CHANGE = 10

class MutationStats():
  def __init__(self, failures, prune_rejections, structural_rejections, seen_rejections, mutation_info):
    self.failures = failures
    self.prune_rejections = prune_rejections
    self.structural_rejections = structural_rejections
    self.seen_rejections = seen_rejections
    self.used_mutation = mutation_info

class MutationInfo():
  def __init__(self):
    self.mutation_type = None
    self.insert_ops = None
    self.binop_operand_choice = None
    self.node_id = -1
    self.new_output_channels = -1
    self.new_grouping = -1
    self.green_model_id = -1



"""
mutates the given tree 
50/50 percent chance of doing an insertion or deletion mutation
unless tree size exceeds MAX_NODES. If tree size exceeds MAX_NODES 
only performs deletion mutation
"""
class Mutator():
  def load_mutation_types_cdf(self, search_mutation_type_pdfs):
    if self.args.full_model:
      if self.args.nas:
        if self.args.insertion_bias:
          self.early_mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["nas"]["insertion_bias"]["early"])
          self.late_mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["nas"]["insertion_bias"]["late"])
        else:
          self.mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["nas"]["uniform"])
      else:
        if self.args.insertion_bias:
          self.early_mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["full_model"]["insertion_bias"]["early"])
          self.late_mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["full_model"]["insertion_bias"]["late"])
        else:
          self.mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["full_model"]["uniform"])
    elif self.args.rgb8chan:
      if self.args.insertion_bias:
        self.early_mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["rgb8chan"]["insertion_bias"]["early"])
        self.late_mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["rgb8chan"]["insertion_bias"]["late"])
      else:
        self.mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["rgb8chan"]["uniform"])
    else:
      if self.args.insertion_bias:
        self.early_mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["green_model"]["insertion_bias"]["early"])
        self.late_mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["green_model"]["insertion_bias"]["late"])
      else:
        self.mutation_type_cdf = np.cumsum(search_mutation_type_pdfs["green_model"]["uniform"])

  def __init__(self, args, debug_logger, mysql_logger):
    self.args = args
    self.debug_logger = debug_logger
    self.mysql_logger = mysql_logger
    
    self.seen_models = {}
    self.seen_structures = set()

    self.failed_mutation_info = []

    self.load_mutation_types_cdf(mutation_types_pdfs)

    if self.args.binop_change: # cdf for whether binop operand is taken from mutating subtree or partner tree or input ops
      self.binop_operand_cdf = [0.20, 0.80, 1.0]

    self.binop_tries = 0
    self.binop_success = 0
    self.binop_insertion_failures = 0
    self.binop_loc_selection_failures = 0
    self.binop_rejects = 0
    self.binop_seen = 0
    self.partner_loc_fails = 0

  def add_seen_model(self, tree, model_id):
    self.seen_models[tree] = {"model_ids":[int(model_id)], "hash": hash(tree), "ast_str": tree.dump()}

  def give_up(self, structural_rejections, prune_rejections, seen_rejections, failures):
    total_tries = structural_rejections + prune_rejections + seen_rejections + failures 
    if total_tries > self.args.mutation_failure_threshold:
      return True      
    else:
      return False      

  def pick_mutation_type(self, tree, generation):
    tree_size = tree.compute_size(set(), count_all_inputs=True)
    if self.args.full_model:
      min_tree_size = 5
    else:
      min_tree_size = 3
    if tree_size > self.args.max_nodes:
      mutation_type = MutationType.DELETION
    elif tree_size <= min_tree_size:
      mutation_type = MutationType.INSERTION
    else:
      if self.args.insertion_bias:
        if generation <= self.args.late_cdf_gen:
          mutation_type_cdf = self.early_mutation_type_cdf
        else:
          mutation_type_cdf = self.late_mutation_type_cdf
      else:
        mutation_type_cdf = self.mutation_type_cdf

      rv = random.uniform(0,1)
      for i, mtype in enumerate(list(MutationType)):
        if rv <= mutation_type_cdf[i]:
          mutation_type = mtype
          break
      # self.debug_logger.debug(f"--- mutation type cdf: {mutation_type_cdf} ---")
    return mutation_type

  def mutate(self, parent_id, model_id, tree, input_set, generation, partner_ast=None):
    self.debug_logger.debug(f"---- ENTERING MUTATION of {parent_id} ----")
    failures = 0
    prune_rejections = 0
    structural_rejections = 0
    seen_rejections = 0
    
    self.failed_mutation_info = []

    while True: # loop to keep trying over assertion errors
      if self.give_up(structural_rejections, prune_rejections, seen_rejections, failures):
        stats = MutationStats(failures, prune_rejections, structural_rejections, seen_rejections, self.current_mutation_info)
        self.debug_logger.debug(f"giving up mutation to produce model {model_id}")
        return None, None, stats

      try:
        while True: # loop to keep trying over tree pruning 
          mutation_type = self.pick_mutation_type(tree, generation)
          try:
            tree_copy = copy_subtree(tree)
            if self.args.binop_change:
              partner_copy = copy_subtree(partner_ast)
          except AssertionError:
            self.debug_logger.debug(f"failed to copy model {parent_id}")
            failures += 1
            continue

          self.current_mutation_info = MutationInfo()
          self.current_mutation_info.mutation_type = mutation_type.name

          #self.debug_logger.debug(f"the tree before mutation:\n{tree_copy.dump()}")
          if mutation_type is MutationType.INSERTION:  
            self.debug_logger.debug(f"attempting insertion")
            if self.args.binop_change:
              new_tree = self.insert_mutation(tree_copy, input_set, partner_ast=partner_copy)
            else:
              new_tree = self.insert_mutation(tree_copy, input_set)
          elif mutation_type is MutationType.DELETION:
            self.debug_logger.debug(f"attempting deletion")
            new_tree = self.delete_mutation(tree_copy)
          elif mutation_type is MutationType.DECOUPLE:
            self.debug_logger.debug(f"attempting decouple")
            new_tree = self.decouple_mutation(tree_copy)
          elif mutation_type is MutationType.CHANNEL_CHANGE:
            self.debug_logger.debug(f"attempting channel change")
            with open("attempts.txt", "a+") as f:
              f.write("1\n")
            new_tree = self.channel_mutation(tree_copy)
          elif mutation_type is MutationType.GROUP_CHANGE:
            self.debug_logger.debug(f"attempting group mutation")
            new_tree = self.group_mutation(tree_copy)
          elif mutation_type is MutationType.INSERT_RESOLUTION_SUBGRAPH:
            self.debug_logger.debug(f"attempting insert resolution subgraph")
            new_tree = self.insert_resolution_subgraph_mutation(tree_copy)
          elif mutation_type is MutationType.DELETE_RESOLUTION_SUBGRAPH:
            self.debug_logger.debug(f"attempting delete resolution subgraph")
            new_tree = self.delete_resolution_subgraph_mutation(tree_copy)
          elif mutation_type is MutationType.SHIFT_RESOLUTION_SUGRAPH:
            self.debug_logger.debug(f"attempting shift resolution subgraph")
            new_tree = self.shift_resolution_subgraph_mutation(tree_copy)
          elif mutation_type is MutationType.SWAP_RESOLUTION_OP:
            self.debug_logger.debug(f"attempting swap resolution op")
            new_tree = self.swap_resolution_op_mutation(tree_copy)
          else:
            self.debug_logger.debug(f"attempting green model change mutation")
            new_tree = self.green_model_change_mutation(tree_copy)

          self.debug_logger.debug(f"--- finished mutation ---")
          normalize.fix_nonlinear_adjacency(new_tree)

          tree_accepted, accept_err = self.accept_tree(new_tree)

          if tree_accepted: 
            break      
          else:
            if mutation_type is MutationType.CHANNEL_CHANGE:
              with open("rejected.txt", "a+") as f:
                f.write("1\n")
          # tree was pruned
          self.failed_mutation_info.append(self.current_mutation_info)
          prune_rejections += 1

      # mutation failed
      except (AssertionError, AttributeError, TypeError) as e: 
        print(f'exception thrown: {e}')
        with open("exceptions.txt", "a+") as f:
          f.write(f"{e}\n\n")
        print(traceback.print_exc())
        if mutation_type is MutationType.INSERTION:
          self.debug_logger.debug(f'insertion mutation failed on parent model {parent_id} tried to insert {self.current_mutation_info.insert_ops}')
        elif mutation_type is MutationType.DELETION:
          self.debug_logger.debug(f'deletion mutation failed on parent model {parent_id}')
        elif mutation_type is MutationType.DECOUPLE:
          self.debug_logger.debug(f'decouple mutation failed on parent model {parent_id}')
        elif mutation_type is MutationType.CHANNEL_CHANGE:
          self.debug_logger.debug(f'channel change mutation failed on model {parent_id}')
          with open("fails.txt", "a+") as f:
            f.write("1\n")  
        elif mutation_type is MutationType.GREEN_MODEL_CHANGE:
          self.debug_logger.debug(f'green model change mutation failed on model {parent_id}')
        elif mutation_type is MutationType.INSERT_RESOLUTION_SUBGRAPH:
          self.debug_logger.debug(f'insert resolution subgraph mutation failed on model {parent_id}')
        elif mutation_type is MutationType.DELETE_RESOLUTION_SUBGRAPH:
          self.debug_logger.debug(f'delete resolution subgraph mutation failed on model {parent_id}')
        else:
          self.debug_logger.debug(f'group mutation failed on model {parent_id}')

        self.failed_mutation_info.append(self.current_mutation_info)
        failures += 1
        continue
      else: # successfully mutated tree
        try:
          print(f"the tree {new_tree.dump()}")
          check_channel_count(new_tree) # these should not fire...
        except AssertionError as e:
          self.debug_logger.debug(f"channel count check failed on model {model_id} the assertion error {e}")
          self.failed_mutation_info.append(self.current_mutation_info)
          if mutation_type is MutationType.CHANNEL_CHANGE:
            with open("channel_check_fails.txt", "a+") as f:
              f.write("1\n")  

          failures += 1
          continue
        try:
          assert_no_nonlinear_adjacency(new_tree) # allow adjacent convs, just no adjacent relu / softmax
        except AssertionError:
          self.debug_logger.debug(f"check no adjacent nonlinear failed on model {model_id}")
          self.failed_mutation_info.append(self.current_mutation_info)
          if mutation_type is MutationType.CHANNEL_CHANGE:
            with open("linear_adjacency_fails.txt", "a+") as f:
              f.write("1\n")  
          failures += 1
          continue

        # reject trees we've seen already 
        if (not new_tree in self.seen_models):
          # reject with some chance trees with previously seen structure
          h = structural_hash(new_tree)
          if h in self.seen_structures:
            if random.random() < self.args.structural_sim_reject:
              structural_rejections += 1
              if mutation_type is MutationType.CHANNEL_CHANGE:
                with open("structural_rejections.txt", "a+") as f:
                  f.write("1\n")
              continue
            else: # accepting tree that is structurally similar to one already seen
              self.seen_models[new_tree] = {"model_ids":[int(model_id)], "hash": hash(new_tree), "ast_str": new_tree.dump()}
              stats = MutationStats(failures, prune_rejections, structural_rejections, seen_rejections, self.current_mutation_info)
              return new_tree, h, stats
          else: # successfully mutated tree!
            self.seen_structures.add(h)
            self.seen_models[new_tree] = {"model_ids":[int(model_id)], "hash": hash(new_tree), "ast_str": new_tree.dump()}
            stats = MutationStats(failures, prune_rejections, structural_rejections, seen_rejections, self.current_mutation_info)
            return new_tree, h, stats

        else: # we've seen and evaluated this exact tree before
          if mutation_type is MutationType.CHANNEL_CHANGE:
            with open("seen_rejections.txt", "a+") as f:
              f.write("1\n")
          self.seen_models[new_tree]["model_ids"].append(int(model_id))
          seen_rejections += 1
          continue
      
def collect_green_nodes(tree):
  green_nodes = []
  nodes = tree.preorder()
  for n in nodes:
    if type(n) is Input:
      if n.name == "Input(GreenExtractor)":
        green_nodes += [n]
      elif hasattr(n, 'node'):
        green_nodes += collect_green_nodes(n.node)
  return green_nodes


@extclass(Mutator)
def green_model_change_mutation(self, tree):
  green_nodes = []
  green_model_id = None

  nodes = tree.preorder()

  green_model_id = get_green_model_id(tree)
  green_nodes = collect_green_nodes(tree)

  green_model_options = [i for i in range(len(self.args.green_model_asts))]
  green_model_options.remove(green_model_id)
  new_green_model_id = random.choice(green_model_options)
  for n in green_nodes:
    n.green_model_id = new_green_model_id

  return tree


"""
randomly chooses a node with multiple parents and splits it 
off from the shared computation, making its child the new
root of the shared computation
"""
@extclass(Mutator)
def decouple_mutation(self, tree, chosen_node_id=None):
  preorder_nodes = tree.preorder()
  if chosen_node_id:
    chosen_node = preorder_nodes[chosen_node_id]
  else:
    shared = list(filter(lambda x: type(x.parent) is tuple and not isinstance(x, Input), preorder_nodes))
    if len(shared) == 0:
      self.debug_logger.debug("cannot perform decouple mutation on a tree with no shared computation")
      assert False, "cannot perform decouple mutation on a tree with no shared computation"
    chosen_node = random.sample(shared, 1)[0]

    for i, n in enumerate(preorder_nodes):
      if n is chosen_node:
        chosen_node_id = i
        break
  self.current_mutation_info.node_id = chosen_node_id

  copies = [copy.copy(chosen_node) for p in chosen_node.parent]
  if chosen_node.num_children == 3:
    children = [chosen_node.child1, chosen_node.child2, chosen_node.child3]
  elif chosen_node.num_children == 2:
    children = [chosen_node.lchild, chosen_node.rchild]
  else:
    children = [chosen_node.child]

  for child in children:
    if type(child.parent) is tuple:
      # remove the chosen node from list of parents and replace it with its copies
      new_parent_tuple = tuple(filter(lambda x: id(x) != id(chosen_node), child.parent)) 
      new_parent_tuple = new_parent_tuple + tuple(copies)
      child.parent = new_parent_tuple
    else:
      child.parent = tuple(copies)
  
  # update copies of the node to each point to one of the parents separately
  # update each of those parents to only point to the one copy it was assigned
  chosen_node_parents = chosen_node.parent
  for i, p in enumerate(chosen_node_parents):
    copies[i].parent = p 
    if p.num_children == 3:
      if chosen_node is p.child1:
        p.child1 = copies[i]
      elif chosen_node is p.child2:
        p.child2 = copies[i]
      else:
        p.child3 = copies[i]
    elif p.num_children == 2:
      if chosen_node is p.lchild:
        p.lchild = copies[i]
      else:
        p.rchild = copies[i]
    else:
      p.child = copies[i]

  return tree


"""
randomly chooses to go up or down to the channel 
option closest to the given output channel 
"""
def perturb_channel(out_c):
  decrease_c = None
  increase_c = None
  for i, option in enumerate(channel_options):
    if option > out_c:
      increase_c = channel_options[i]
      break
  for i, option in enumerate(reversed(channel_options)):
    if option < out_c:
      decrease_c = channel_options[len(channel_options)-i-1]

  if decrease_c is None:
    return increase_c
  elif increase_c is None:
    return decrease_c
  else:
    go_down = random.randint(0,1)
    if go_down:
      return decrease_c
    else:
      return increase_c

"""
channel count mutation 
changes channel count of randomly chosen conv 
"""
@extclass(Mutator)
def channel_mutation(self, tree, chosen_conv_id=None):
  preorder_nodes = tree.preorder()

  if chosen_conv_id:
    chosen_conv = preorder_nodes[chosen_conv_id]
  else:
    candidate_nodes = list(filter(lambda x: type(x) in ops_with_changeable_channels, preorder_nodes))
    if len(candidate_nodes) == 0:
      self.debug_logger.debug("cannot perform channel mutation, tree has no nodes with changeable channels")
      assert False, "cannot perform channel mutation, tree has no nodes with changeable channels"

    chosen_node = random.choice(candidate_nodes)

    for i, n in enumerate(preorder_nodes):
      if n is chosen_node:
        chosen_node_id = i
        break

  new_out_c = perturb_channel(chosen_node.out_c)

  self.current_mutation_info.node_id = chosen_node_id
  self.current_mutation_info.new_output_channels = new_out_c

  fixed = fix_channel_count_upwards(chosen_node, new_out_c)
  if fixed:
    chosen_node.out_c = new_out_c
    # if the mutated convolution's new output channels is not divisible by its groups,
    # set the groups to the factor of its in and out channels closest to its current groups
    in_c_factors = get_factors(chosen_node.in_c)
    if type(chosen_node) is Conv1D:
      out_c = chosen_node.out_c // 2
    else:
      out_c = chosen_node.out_c 

    out_c_factors = get_factors(out_c)
    factors = in_c_factors.intersection(out_c_factors)
    closest_factor = get_closest_factor(factors, chosen_node.groups)
    chosen_node.groups = closest_factor # if current groups is already a factor, this does nothing
    tree.compute_input_output_channels()
    return tree
  else:
    self.debug_logger.debug(f"Unable to perturb output channel of \n{chosen_node.dump()}\n from {chosen_node.out_c} to {new_out_c}")
    print(f"in tree\n{tree.dump()}")
    assert False, f"Unable to perturb channel counts of node {chosen_node}"


"""
grouped conv mutation 
changes grouping factor of randomly chosen conv
"""
@extclass(Mutator)
def group_mutation(self, tree):
  preorder_nodes = tree.preorder()
  convs = list(filter(lambda x: type(x) in linear_ops.union(special_linear_ops), preorder_nodes)) 
  if len(convs) == 0:
    self.debug_logger.debug("cannot perform grouped conv mutation on tree with no convs")
    assert False, "cannot perform grouped conv mutation on tree with no convs"

  chosen_conv = random.sample(convs, 1)[0]
  for i, n in enumerate(preorder_nodes):
    if n is chosen_conv:
      chosen_conv_id = i
      break
  self.current_mutation_info.node_id = chosen_conv_id

  if type(chosen_conv) is Conv1D:
    out_c = chosen_conv.out_c // 2
  else:
    out_c = chosen_conv.out_c

  in_c_factors = get_factors(chosen_conv.in_c)
  out_c_factors = get_factors(out_c)
  factors = in_c_factors.intersection(out_c_factors)
  factors.remove(chosen_conv.groups)
  if len(factors) == 0:
    self.debug_logger.debug("No possible grouping factors for grouped conv mutation")
    assert False, "No possible grouping factors for grouped conv mutation"
  # randomly pick a factor to use as the group 
  new_grouping = random.sample(factors, 1)[0]
  
  self.current_mutation_info.new_grouping = new_grouping

  chosen_conv.groups = new_grouping # input and output channels remain unchanged 
  return tree
  

"""
returns whether parent and child linearity type sequence is allowed
"""
@extclass(Mutator)
def legal_parent_child_linearity(self, parent, child):
  if type(parent) is tuple:
    parents = list(parent)
  else:
    parents = [parent]
  
  child_type = type(child)

  illegal_parents = []
  for p in parents:
    parent_type = type(p)
    # parent_child_ok = (parent_type in border_ops or child_type in border_ops \
    #       or (parent_type in nl_and_sp and child_type in l_and_sp) \
    #       or (parent_type in l_and_sp and child_type in nl_and_sp))
    """
    allowing adjacent linear ops now
    """
    bad_parent_child = (parent_type in nonlinear_ops) and (child_type in nonlinear_ops)

    if bad_parent_child: 
      illegal_parents += [p]

  return illegal_parents


"""
returns whether a node still exists in the tree
"""
def is_in_tree(tree, node):
  preorder_nodes = tree.preorder()
  for n in preorder_nodes:
    if id(n) == id(node):
      return True
  return False


@extclass(Mutator)
def insert_resolution_subgraph_mutation(self, tree):
  possible_factors = self.args.resolution_change_factors
  # chosen_factor = random.choice(possible_factors)
  change_subgraph_resolution(tree, possible_factors, self.args.pixel_width, MAX_TRIES=self.args.max_resolution_change_tries)
  return tree

@extclass(Mutator)
def delete_resolution_subgraph_mutation(self, tree):
  delete_resolution_subgraph(tree, self.args.pixel_width)
  return tree

@extclass(Mutator)
def shift_resolution_subgraph_mutation(self, tree):
  flip_resolution(tree, self.args.pixel_width)
  return tree

@extclass(Mutator)
def swap_resolution_op_mutation(self, tree):
  swap_resolution_op(tree, self.args.pixel_width)
  return tree


"""

Functions for deletion

"""

"""
given a dictionary of ids to nodes that have been removed down one path 
of the DAG, (but may still be reachable through other parents still in the DAG)
remove connections to parents that are no longer reachable 
"""
def remove_connections_to_deleted_nodes(tree, deleted_nodes):
  for nid, n in deleted_nodes.items():
    if type(n.parent) is tuple:
      parents = list(n.parent)
    else:
      parents = [n.parent]

    reachable_parents = []
    for p in parents:
      if is_in_tree(tree, p):
        reachable_parents.append(p)

    if len(reachable_parents) > 1:
      n.parent = tuple(reachable_parents)
    elif len(reachable_parents) == 1:
      n.parent = reachable_parents[0]
    else:
      n.parent = None

"""
fixes up parent child relationships to remove given node
"""
def remove_node(parent, deletion_node, new_child):
  replace_child(parent, deletion_node, new_child)
  replace_parent(new_child, deletion_node, parent)


"""
fixes channel counts after deleting nodes
"""
@extclass(Mutator)
def fix_channels_after_deletion(self, tree, deletion_parent, deletion_child):
  if deletion_parent is None:
    return 

  if type(deletion_parent) is tuple: # must fix upwards if deletion child has mulitple parents
    deletion_child.compute_input_output_channels()
    fixed = fix_channel_count_upwards(deletion_child, deletion_child.out_c)
    if fixed:
      tree.compute_input_output_channels()
      check_channel_count(tree)
    else:
      self.debug_logger.debug("Could not make channel counts agree after deleting ops")
      assert False, "Could not make channel counts agree after deleting ops"
  else: # deletion child only has one parent
    if deletion_parent.num_children == 3:
      if id(deletion_child) == id(deletion_parent.child1):
        out_c = deletion_parent.in_c[0]
      elif id(deletion_child) == id(deletion_parent.child2):
        out_c = deletion_parent.in_c[1]
      else:
        out_c = deletion_parent.in_c[2]
    elif deletion_parent.num_children == 2:
      if id(deletion_child) == id(deletion_parent.lchild):
        out_c = deletion_parent.in_c[0]
      else:
        out_c = deletion_parent.in_c[1]
    else:
      out_c = deletion_parent.in_c

    # try to fix channels downwards, keep track of original input output channels 
    # so we can reset them and try fixing upwards if fixing downwards fails
    in_out_channels = deletion_child.get_input_output_channels()
    fixed = fix_channel_count_downwards(deletion_child, deletion_parent, out_c)
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
        self.debug_logger.debug("Could not make channel counts agree after deleting ops")
        assert False, "Could not make channel counts agree after deleting ops"


"""
deletes nodes from tree starting from the given node up the tree until
linearity rules are obeyed.
reassigns deleted subtrees to one if any of its dependees
returns the child of the deleted nodes
"""
@extclass(Mutator)
def delete_nodes(self, tree, node):
  # if id(node) in already_deleted and not is_in_tree(tree, node):
  #   return tree
  if not is_in_tree(tree, node):
    return tree

  parent = node.parent
  deleted_nodes = {}

  self.debug_logger.debug(f"deleting {id(node)} {node.name}")

  deleted_nodes[id(node)] = node

  if isinstance(node, Binop):
    # decide if you're going to keep left or right child
    keep_left = random.randint(0,1)
    if keep_left: 
      child = node.lchild
      for dn in node.rchild.preorder():
        deleted_nodes[id(dn)] = dn 
    else:
      child = node.rchild
      for dn in node.lchild.preorder():
        deleted_nodes[id(dn)] = dn

  else: # deleting a Unop
     child = node.child

  if type(parent) is tuple:
    for p in parent:
      remove_node(p, node, child)
  else:
    remove_node(parent, node, child)

  self.fix_channels_after_deletion(tree, parent, child)

  illegal_parents = self.legal_parent_child_linearity(parent, child)

  remove_connections_to_deleted_nodes(tree, deleted_nodes)

  return tree


"""
picks a node to delete from a tree - ok as long as not a border node or resolution op
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
    if not (type(node) in border_ops or isinstance(node, Downsample) or isinstance(node, Upsample)):
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

  self.debug_logger.debug(f"the chosen delete node {node} id {tree.get_preorder_id(node)} in tree\n{tree.dump()}")
  self.current_mutation_info.node_id = tree.get_preorder_id(node)
  tree = self.delete_nodes(tree, node)
  tree.compute_input_output_channels()
  return tree 




"""

functions for supporting insertion 

"""
@extclass(Mutator)
def insert_binop(self, tree, input_set, insert_op, insert_parent, insert_child, make_copy=False, partner_ast=None):
  compute_resolution(tree)

  OpClass = insert_op
  if not hasattr(tree, "size"):
    tree.compute_size(set(), count_all_inputs=True)

  if tree.size < self.args.min_subtree_size + 2:
    self.debug_logger.debug(f"Impossible to find subtree of size {self.args.min_subtree_size} or greater from tree of size {tree.size}")
    assert False, f"Impossible to find subtree of size {self.args.min_subtree_size} or greater from tree of size {tree.size}"

  #### pick a subtree ####
  if issubclass(OpClass, BinopIII):
    # make subtree's output channels and spatial resolution match other child's output channels
    subtree = self.pick_subtree(tree, input_set, insert_child, OpClass, target_out_c=insert_child.out_c, resolution=insert_child.resolution, make_copy=make_copy, partner_ast=partner_ast) 
  else: # Op is BinopIJK - keep output channels of the chosen subtree unchanged, make spatial resolution match insert child's
    subtree = self.pick_subtree(tree, input_set, insert_child, OpClass, resolution=insert_child.resolution, make_copy=make_copy, partner_ast=partner_ast)
  
  # check spatial resolutions of subtree and insert_child match 
  subtree_res = subtree.resolution
  insert_child_res = insert_child.resolution

  if subtree_res != insert_child_res:
    # TODO: ALLOW INSERTION OF DOWNSAMPLE OR UPSAMPLE TO MAKE RESOLUTIONS MATCH ?
    self.debug_logger.debug(f"resolution of chosen subtree for binary op does not match insert_child resolution")
    assert False, f"resolution of chosen subtree for binary op does not match insert_child resolution"
   
  use_right = random.randint(0,1)
 
  if use_right:
    new_node = OpClass(insert_child, subtree, resolution=subtree_res)
  else:
    new_node = OpClass(subtree, insert_child, resolution=subtree_res)

  # parent is not none when inserting a required right child OR not making a copy of the chosen subtree
  insertion_edge_updates(insert_parent, insert_child, new_node)
  insertion_edge_updates(insert_parent, subtree, new_node)

  return new_node


"""
Inserts a unary op
resets insert_child's parent to point to newly created node
new node points to insert_child as its child
INSERT PARENT IS NOT MODIFIED TO POINT TO THE NEW NODE
"""
@extclass(Mutator)
def insert_unary_op(self, OpClass, insert_parent, insert_child):
  resolution = insert_child.resolution
  if issubclass(OpClass, UnopIIdiv):
    in_c_factors = get_factors(insert_child.out_c)
    # remove the input channels from the list of output channel options
    in_c_factors.remove(insert_child.out_c)
    assert len(in_c_factors) > 0, f"Cannot insert {OpClass} without output channels to choose from"
    out_c = random.sample(in_c_factors, 1)[0]
    new_node = OpClass(insert_child, out_c, resolution=resolution)
  elif issubclass(OpClass, UnopIJ):
    out_c = insert_child.out_c
    new_node = OpClass(insert_child, out_c, resolution=resolution)    
  else:
    assert issubclass(OpClass, UnopII), f"Invalid Unop type {OpClass}"
    new_node = OpClass(insert_child, resolution=resolution)
 
  insertion_edge_updates(insert_parent, insert_child, new_node)

  return new_node


"""
checks if insertion location is ok for the given op, asserts False if not ok.
if OK, returns any necessary additional ops to keep nonlin / lin op adjacency
"""
def accept_insertion_loc(child, parent, insert_op):
  # don't allow insertion below a border op - this is only a problem with chroma prediction
  # where we have Flat2Quad above an Input node 
  if type(parent) in border_ops:
    assert False, "rejecting insert location upstream from border op {type(parent)}"

  # reject two UnopIIdivs in a row (i.e. InterleavedSum / GroupedSum)
  if issubclass(insert_op, UnopIIdiv):
    if isinstance(parent, UnopIIdiv) or isinstance(child, UnopIIdiv):
      assert False, "rejecting insert location for UnopIIdiv next to another UnopIIdiv"

  if insert_op is Mul:
    if isinstance(parent, Mul) or isinstance(child, Mul):
      assert False, "rejecting insert location for Mul next to another Mul"

  if isinstance(parent, Linear):
    if isinstance(child, NonLinear):
      if insert_op in linear_insert_ops:
        nonlin_op = random.sample(nonlinear_insert_ops, 1)[0]
        return [nonlin_op, insert_op]
      elif insert_op in nonlinear_insert_ops:
        lin_op = random.sample(linear_insert_ops, 1)[0]
        return [insert_op, lin_op]
      else:
        return [insert_op] # op is special op 
    elif isinstance(child, Special): # parent is linear, child is special
      if insert_op in linear_insert_ops:
        nonlin_op = random.sample(nonlinear_insert_ops, 1)[0]
        return [nonlin_op, insert_op]
      else:
        return [insert_op]
    else: # parent is Linear, child is Linear
      if insert_op in linear_insert_ops:
        nonlin_op1 = random.sample(nonlinear_insert_ops, 1)[0]
        nonlin_op2 = random.sample(nonlinear_insert_ops, 1)[0]
        return [nonlin_op1, insert_op, nonlin_op2]
      else:
        return [insert_op]
  elif isinstance(parent, NonLinear):
    if isinstance(child, Linear):
      if insert_op in linear_insert_ops:
        nonlin_op = random.sample(nonlinear_insert_ops, 1)[0]
        return [insert_op, nonlin_op]
      elif insert_op in nonlinear_insert_ops:
        lin_op = random.sample(linear_insert_ops, 1)[0]
        return [lin_op, insert_op]
      else:
        return [insert_op]
    elif isinstance(child, Special):
      if insert_op in nonlinear_insert_ops:
        lin_op = random.sample(linear_insert_ops, 1)[0]
        return [lin_op, insert_op]
      else:
        return [insert_op]
    else: # parent is Nonlinear, child is NonLinear
      if insert_op in nonlinear_insert_ops:
        lin_op1 = random.sample(linear_insert_ops, 1)[0]
        lin_op2 = random.sample(linear_insert_ops, 1)[0]
        return [lin_op1, insert_op, lin_op2]
      else:
        return [insert_op]
  elif isinstance(parent, Special):
    if isinstance(child, Linear):
      if insert_op in linear_insert_ops:
        nonlin_op = random.sample(nonlinear_insert_ops, 1)[0]
        return [insert_op, nonlin_op]
      else:
        return [insert_op]
    elif isinstance(child, NonLinear):
      if insert_op in nonlinear_insert_ops:
        lin_op = random.sample(linear_insert_ops, 1)[0]
        return [insert_op, lin_op]
      else:
        return [insert_op]
    else: # parent and child are both special
      return [insert_op]


"""
Chooses which activation function to insert under a Mul
"""
def choose_activation_under_mul(mul_node):
  use_relu = random.randint(0,1)
  use_left = random.randint(0,1)
  if use_relu:
    insert_ops = [Relu]
  else:
    insert_ops = [Softmax]
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
  insert_nodes = None
  if len(relu_or_softmax.intersection(childtypes)) == 0:
    insert_op, insert_child = choose_activation_under_mul(mul_node)
    insert_nodes, tree = self.insert(tree, insert_op, mul_node, insert_child, input_set)
  return insert_nodes, tree


def get_out_c_from_parent(child, parent):
  if parent.num_children == 3:
    children = [parent.child1, parent.child2, parent.child3]
    out_c = [parent.in_c[0], parent.in_c[1], parent.in_c[2]]
  elif parent.num_children == 2:
    children = [parent.lchild, parent.rchild]
    out_c = [parent.in_c[0], parent.in_c[1]]
  else:
    children = [parent.child]
    out_c = [parent.in_c]
  for i, c in enumerate(children):
    if id(c) == id(child):
      return out_c[i]


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
def insert(self, tree, insert_ops, insert_parent, insert_child, input_set, partner_ast=None):
  cur_child = insert_child
  new_nodes = [] 

  for i, OpClass in enumerate(reversed(insert_ops)):
    cur_child.compute_input_output_channels()
    if issubclass(OpClass, Binop):
      params = list(signature(OpClass).parameters.items())
      assert len(params) == 4, f"Invalid number of parameters {len(params)} for Binary op"
      new_node = self.insert_binop(tree, input_set, OpClass, insert_parent, cur_child, partner_ast=partner_ast)
     
    elif issubclass(OpClass, Unop):
      new_node = self.insert_unary_op(OpClass, insert_parent, cur_child)

    new_nodes = [new_node] + new_nodes
    cur_child = new_node

  # make sure output channels of inserted nodes agree with parent
  cur_child.compute_input_output_channels()

  cur_child_id = tree.get_preorder_id(cur_child)
  insert_parent_id = tree.get_preorder_id(insert_parent)

  tree_copy = copy_subtree(tree) # make a copy in case fixing upwards fails, we can try fixing down
  copy_nodes = tree_copy.preorder()
  new_node_copies = [copy_nodes[tree.get_preorder_id(n)] for n in new_nodes]

  fixed = fix_channel_count_upwards(cur_child, cur_child.out_c)
  if not fixed:
    cur_child_copy = copy_nodes[cur_child_id]
    insert_parent_copy = copy_nodes[insert_parent_id]
  
    output_c = get_out_c_from_parent(cur_child_copy, insert_parent_copy)

    fixed = fix_channel_count_downwards(cur_child_copy, insert_parent_copy, output_c)
    if fixed:
      tree = tree_copy
      new_nodes = new_node_copies
    else:
      self.debug_logger.debug("unable to make inserted nodes channel counts agree with tree")
      assert False, "Could not make channel counts agree with inserted ops"

  tree.compute_input_output_channels()

  check_channel_count(tree)
  return new_nodes, tree


@extclass(Mutator)
def insert_mutation(self, tree, input_set, insert_above_node_id=None, insert_op=None, partner_ast=None):
  preorder_nodes = tree.preorder()

  while True:
    # pick an op to insert
    if not insert_op:
      if self.args.nas:
        insert_op = random.sample(nas_ops, 1)[0]
      else:
        insert_op = random.sample(all_insert_ops, 1)[0]
      if insert_op is Add or insert_op is Sub or insert_op is Stack or insert_op is Mul:
        self.binop_tries += 1

    location_selection_failures = 0 

    while location_selection_failures < self.args.insert_location_tries:
      if not insert_above_node_id:
        insert_above_node_id = random.randint(1, len(preorder_nodes)-1)

      insert_child = preorder_nodes[insert_above_node_id]
      insert_parent = insert_child.parent

      if type(insert_parent) is tuple:
        insert_parent = random.choice(list(insert_parent))
      try:
        insert_ops = accept_insertion_loc(insert_child, insert_parent, insert_op)
                
      except AssertionError:
        insert_above_node_id = None
        location_selection_failures += 1

        if location_selection_failures >= self.args.insert_location_tries:
          if insert_op is Add or insert_op is Sub or insert_op is Stack or insert_op is Mul:
            self.binop_loc_selection_failures += 1
          insert_op = None 

        continue
          
      self.current_mutation_info.insert_ops = ";".join([new_op.__name__ for new_op in insert_ops])
      self.current_mutation_info.node_id = insert_above_node_id

      try:
        new_nodes, tree = self.insert(tree, insert_ops, insert_parent, insert_child, input_set, partner_ast=partner_ast)
      except AssertionError:
        if insert_op is Add or insert_op is Sub or insert_op is Stack or insert_op is Mul:
          self.binop_insertion_failures += 1
        assert False, "insertion mutation failed"
      
      """
      if we inserted Mul, and neither child is Softmax or Relu, add Softmax or Relu as a child
      """
      for OpClass, new_node in zip(insert_ops, new_nodes):
        if OpClass is Mul:
          _, tree = self.insert_activation_under_mul(tree, input_set, new_node)
      
      if insert_op is Add or insert_op is Sub or insert_op is Stack or insert_op is Mul:
        self.binop_success += 1

      return tree


def has_downsample(tree):
  if isinstance(tree, Downsample):
    return True
  if tree.num_children == 3:
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
  if tree.num_children == 3:
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
returns whether or not the chosen subtree will induce a loop
"""
def induces_loop(insert_child, subtree):
  if id(insert_child) == id(subtree):
    return True
  if insert_child.parent is None:
    return False 
  parent = insert_child.parent
  if type(parent) is tuple:
    for p in parent:
      if id(p) == id(subtree):
        return True
      if induces_loop(p, subtree):
        return True
  else:
    if id(parent) == id(subtree):
      return True
    return induces_loop(parent, subtree)
  return False

  
"""
checks whether subtree obeys the restricted input set and can be modified 
to agree with given target output channels.
if resolution is given, checks spatial resolution of subtree matches
Returns a copy of the subtree with channels appropriately modified if necessary
"""
@extclass(Mutator)
def allow_subtree(self, root, input_set, insert_child, binop_class, target_out_c=None, resolution=None, make_copy=False):
  if insert_child.is_same_as_wrapper(root) and not binop_class is Mul: # only allow Mul to operate on the same subtrees, Add, Sub, and Stack must use distinct operands 
    self.debug_logger.debug(f"chosen subtree cannot be the same tree as other child of binop")
    assert False, f"chosen subtree cannot be the same tree as other child of binop "

  subtree_resolution = compute_resolution(root)
  if subtree_resolution != resolution:
    self.debug_logger.debug(f"subtree with resolution {subtree_resolution} does not match required resolution {resolution}")
    assert False, f"subtree with resolution {subtree_resolution} does not match required resolution {resolution}"
  size = root.compute_size(set(), count_input_exprs=True)
  # reject small trees or large trees
  if size < self.args.min_subtree_size or size > self.args.max_subtree_size: 
      self.debug_logger.debug(f"Subtree of size {size} is too small or too large")
      assert False, f"Subtree of size {size} is too small or too large"

  # reject trees that require inputs out of sanctioned input_set
  subtree_inputs = root.get_inputs()
  if not subtree_inputs.issubset(input_set):
    self.debug_logger.debug("rejecting subtree with invalid inputs")
    assert False, "rejecting subtree with invalid inputs"
  
  # have to make copy if chosen subtree induces a loop
  # loops occur when chosen subtree is a parent of the insert child
  if induces_loop(insert_child, root):
    self.debug_logger.debug("CHOSEN SUBTREE INDUCES LOOP")

  if make_copy or induces_loop(insert_child, root):
    root = copy_subtree(root)
    root.parent = None

  # reject trees that can't be made to agree with desired output channels
  if target_out_c:
    in_c, out_c = root.compute_input_output_channels()
    if out_c != target_out_c: 
      # if the insert child is downstream of the chosen subtree, we cannot change the subtree's 
      # channels without also affecting the channels of the insert child, resulting in livelock -> must copy subtree
      # if insert_child.is_downstream(root):
      #   root = copy_subtree(root)
      #   root.parent = None

      fixed = fix_channel_count_downwards(root, None, target_out_c)
      # we must also fix upwards now because subtrees are by default not copied,
      # other parent of subtree is affected by channel change
      if fixed: 
        fixed = fix_channel_count_upwards(root, target_out_c)
      if fixed:
        root.compute_input_output_channels()
      else:
        self.debug_logger.debug(f"rejecting subtree: cannot make output channels {out_c} match {target_out_c}")
        assert False, f"rejecting subtree: cannot make output channels {out_c} match {target_out_c}"
  return root
  

"""
selects a subtree from the given tree at root.
Returns a copy of that subtree without any pointers to its parent tree
binop_class is the binop that we're picking a subtree for
"""
@extclass(Mutator)
def pick_subtree(self, root, input_set, insert_child, binop_class, target_out_c=None, resolution=None, root_id=None, make_copy=False, partner_ast=None):
  preorder = root.preorder()
  failures = 0
  while True:
    if root_id and failures == 0:
      subtree_id = root_id
    else:
      if self.args.binop_change:        
        rv = random.uniform(0,1)
        for i, operandtype in enumerate(list(BinopOperandType)):
          if rv <= self.binop_operand_cdf[i]:
            chosen_operand_type = operandtype
            break

        if chosen_operand_type is BinopOperandType.INPUTOP:
          subtree = copy_subtree(random.sample(list(self.args.input_ops), 1)[0])
        elif chosen_operand_type is BinopOperandType.RECURRENT:
          subtree_id = random.randint(1, len(preorder)-1)
          subtree = preorder[subtree_id]
        elif chosen_operand_type is BinopOperandType.CROSSTREE:
          partner_tree_preorder = partner_ast.preorder()
          subtree_id = random.randint(1, len(partner_tree_preorder)-1)
          subtree = partner_tree_preorder[subtree_id]

        self.current_mutation_info.binop_operand_choice = chosen_operand_type.name

      else:
        subtree_id = random.randint(1, len(preorder)-1)
        subtree = preorder[subtree_id]

    subtree.compute_input_output_channels()
    try: 
      chosen_subtree = self.allow_subtree(subtree, input_set, insert_child, binop_class, target_out_c, resolution, make_copy)
    except AssertionError:
      failures += 1
      if failures > self.args.subtree_selection_tries:
        self.debug_logger.debug(f"TOO MANY TRIES TO FIND SUBTREE with resolution {resolution}")
        assert False, f"Too many tries to find subtree with resolution {resolution}"
    else:
      if make_copy:
        chosen_subtree.parent = None
      return chosen_subtree


"""
Pruner
rejects stupid trees
"""
@extclass(Mutator)
def accept_tree(self, tree):

  if tree.parent is None and not tree.has_parameters():
    self.debug_logger.debug(f"rejecting tree without learnable parameters\n {tree.dump()}\n")
    return False, f"rejecting tree without learnable parameters"

  if len(tree.preorder()) > self.args.max_nodes:
    self.debug_logger.debug(f"rejecting tree with size {len(tree.preorder())} larger than max tree size")
    return False, f"rejecting tree with size {len(tree.preorder())} larger than max tree size"

  if compute_resolution(tree) is None:
    self.debug_logger.debug("rejecting invalid spatial resolution")
    return False, "rejecting invalid spatial resolution"
    
  if isinstance(tree, Downsample):
    if isinstance(tree.child, Upsample) or isinstance(tree.child, Downsample):
      self.debug_logger.debug("rejecting adjacent Downsample and Upsample")
      return False, "rejecting adjacent Downsample and Upsample"

  if isinstance(tree, Upsample):
    if isinstance(tree.child, Downsample) or isinstance(tree.child, Upsample):
      self.debug_logger.debug("rejecting adjacent Upsample and Downsample")
      return False, "rejecting adjacent Upsample and Downsample"

  # reject DAGs with receptive fields larger than threshold
  footprint = tree.compute_footprint(1)
  if footprint > self.args.max_footprint:
    self.debug_logger.debug(f"rejecting DAG with footprint {footprint}")
    return False, f"rejecting DAG with footprint {footprint}"

  # reject DAGs that downsample too much 
  min_resolution = min([n.resolution for n in tree.preorder()])
  if min_resolution < self.args.min_resolution:
    self.debug_logger.debug(f"rejecting DAG with min resolution {min_resolution}")
    return False, f"rejecting DAG with min resolution {min_resolution}"

  # reject DAGs with channel counts larger than threshold
  max_channels = get_max_channels(tree)
  if max_channels > self.args.max_channels:
    self.debug_logger.debug(f"rejecting tree with max channels {max_channels}")
    return False, f"rejecting tree with max channels {max_channels}"

  tree_type = type(tree) 
  if tree_type is InterleavedSum or tree_type is GroupedSum: 
    # don't sum reduce over one channel
    if tree.child.out_c == 1:
      self.debug_logger.debug("rejecting sum over one channel")
      return False, "rejecting sum over one channel"
    if type(tree.child) is InterleavedSum and tree_type is InterleavedSum:
      return False, "rejecting adjacent InterleavedSum"
    if type(tree.child) is GroupedSum and tree_type is GroupedSum:
      return False, "rejecting adjacent GroupedSum"

  # don't allow stacking of the same tree
  if tree_type is Stack:
    if tree.lchild.is_same_as_wrapper(tree.rchild):
      self.debug_logger.debug("Rejecting stacking the same tree")
      return False, "Rejecting stacking the same tree"

  # don't allow subtraction or addition of same trees
  if tree_type is Sub or tree_type is Add:
    if tree.lchild.is_same_as_wrapper(tree.rchild):
      self.debug_logger.debug("Rejecting sub or add same tree")
      return False, "Rejecting sub or add same tree"
    # don't allow addition or subtraction of linear subtrees
    if is_linear(tree):
      self.debug_logger.debug("Rejecting sub or add linear trees")
      return False, "Rejecting sub or add linear trees"
  
  # reject invalid groups
  if issubclass(tree_type, Linear):
    if tree_type is Conv1D: 
      out_c = tree.out_c // 2
    else:
      out_c = tree.out_c
    if out_c % tree.groups != 0 or tree.in_c % tree.groups != 0:
      self.debug_logger.debug("rejecting convolution with invalid grouping")
      assert False, "rejecting convolution with invalid grouping"

  if tree_type is Conv1D:
    if tree.out_c % 2 != 0:
      self.debug_logger.debug("rejecting Conv1D with odd output channels")
      assert False, "rejecting Conv1D with odd output channels"
  
  # don't allow softmax over a single channel
  if tree_type is Softmax and tree.out_c == 1:
    return False, "rejecting softmax over one channel"

  # don't allow two softmaxes in same branch
  if tree_type is Softmax:
    ancestors_from_binop_to_node = find_closest_ancestor(tree, set((Binop,)))
    instances = sum([1 for n in ancestors_from_binop_to_node if type(n) is tree_type])
    if instances > 1:
      self.debug_logger.debug("Rejecting multiple softmaxes in the same branch")
      return False, "Rejecting multiple softmaxes in the same branch"

  # Mul must have either Softmax or Relu as one of its children
  # and do not allow two Mul in a row
  if tree_type is Mul:
    if type(tree.rchild) is Mul or type(tree.lchild) is Mul:
      self.debug_logger.debug("rejecting two consecutive Mul")
      return False, "rejecting two consecutive Mul"

    relu_or_softmax = set((Relu, Softmax))
    childtypes = set((type(tree.lchild), type(tree.rchild)))
    if isinstance(tree.lchild, Upsample) or isinstance(tree.lchild, Downsample): # allow resolution changes between Mul and relu or softmax
      childtypes.add(type(tree.lchild.child))
    if isinstance(tree.rchild, Upsample) or isinstance(tree.rchild, Downsample):
      childtypes.add(type(tree.rchild.child))

    if len(relu_or_softmax.intersection(childtypes)) == 0:
      self.debug_logger.debug("rejecting Mul without ReLU or Softmax child")
      return False, "rejecting Mul without ReLU or Softmax child"

  if tree.num_children == 0:
    return True, ""
  elif tree.num_children == 2:
    leftok, lefterr = self.accept_tree(tree.lchild)
    if leftok:
      return self.accept_tree(tree.rchild)
    else:
      return leftok, lefterr
  elif tree.num_children == 1:
    return self.accept_tree(tree.child)
  elif tree.num_children == 3:
    child1ok, child1err = self.accept_tree(tree.child1)
    if child1ok:
      child2ok, child2err = self.accept_tree(tree.child2)
      if child2ok:
        return self.accept_tree(tree.child3)
      else:
        return child2ok, child2err
    else:
      return child1ok, child1err



