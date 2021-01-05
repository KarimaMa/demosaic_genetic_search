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
from util import extclass, get_factors, get_closest_factor
from enum import Enum
from footprint import compute_footprint

channel_options = [8, 10, 12, 16, 20, 24, 28, 32]


class MutationType(Enum):
  DELETION = 1
  INSERTION = 2
  DECOUPLE = 3
  CHANNEL_CHANGE = 4
  GROUP_CHANGE = 5
  GREEN_MODEL_CHANGE = 6

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
  def __init__(self, args, debug_logger, mysql_logger):
    
    self.args = args
    self.debug_logger = debug_logger
    self.mysql_logger = mysql_logger
    
    self.seen_models = {}
    self.seen_structures = set()

    self.failed_mutation_info = []
    
    if self.args.full_model:  
      self.mutation_type_cdf = [0.10, 0.56, 0.66, 0.78, 0.90, 1.0]

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
      if self.args.full_model:
        rv = random.uniform(0,1)
        for i, mtype in enumerate(list(MutationType)):
          if rv <= self.mutation_type_cdf[i]:
            mutation_type = mtype
            break
      else:
        mutation_type = random.choice(list(MutationType)[:-1])
    return mutation_type

  def mutate(self, parent_id, model_id, tree, input_set):
    self.debug_logger.debug(f"---- ENTERING MUTATION of {parent_id} ----")
    failures = 0
    prune_rejections = 0
    structural_rejections = 0
    seen_rejections = 0
    
    self.failed_mutation_info = []

    while True: # loop to keep trying over assertion errors
      try:
        while True: # loop to keep trying over tree pruning 
          mutation_type = self.pick_mutation_type(tree)
          try:
            tree_copy = copy_subtree(tree)
          except AssertionError:
            self.debug_logger.debug(f"failed to copy model {parent_id}")
            failures += 1
            continue

          self.current_mutation_info = MutationInfo()
          self.current_mutation_info.mutation_type = mutation_type.name

          #self.debug_logger.debug(f"the tree before mutation:\n{tree_copy.dump()}")
          if mutation_type is MutationType.INSERTION:  
            self.debug_logger.debug(f"attempting insertion")
            new_tree = self.insert_mutation(tree_copy, input_set)
          elif mutation_type is MutationType.DELETION:
            self.debug_logger.debug(f"attempting deletion")
            new_tree = self.delete_mutation(tree_copy)
          elif mutation_type is MutationType.DECOUPLE:
            self.debug_logger.debug(f"attempting decouple")
            new_tree = self.decouple_mutation(tree_copy)
          elif mutation_type is MutationType.CHANNEL_CHANGE:
            self.debug_logger.debug(f"attempting channel change")
            new_tree = self.channel_mutation(tree_copy)
          elif mutation_type is MutationType.GROUP_CHANGE:
            self.debug_logger.debug(f"attempting group mutation")
            new_tree = self.group_mutation(tree_copy)
          else:
            self.debug_logger.debug(f"attempting green model change mutation")
            new_tree = self.green_model_change_mutation(tree_copy)

          self.debug_logger.debug(f"--- finished mutation ---")

          if self.accept_tree(new_tree):    
            break      

          # tree was pruned
          self.failed_mutation_info.append(self.current_mutation_info)
          prune_rejections += 1

          if self.give_up(structural_rejections, prune_rejections, seen_rejections, failures):
            stats = MutationStats(failures, prune_rejections, structural_rejections, seen_rejections, self.current_mutation_info)
            self.debug_logger.debug(f"giving up mutation to produce model {model_id}")
            return None, None, stats

      # mutation failed
      except (AssertionError, AttributeError, TypeError) as e: 
        if mutation_type is MutationType.INSERTION:
          if "Downsample" in self.current_mutation_info.insert_ops:
            with open(f"{self.args.save}/downsample_fails.txt", "a+") as f:
              f.write(f"failed insert {self.current_mutation_info.insert_ops} above node {self.current_mutation_info.insert_ops.node_id}\n")
              f.write(f"in model{parent_id}\n{tree.dump()}\n--------\n")
          self.debug_logger.debug(f'insertion mutation failed on parent model {parent_id}')
        elif mutation_type is MutationType.DELETION:
          self.debug_logger.debug(f'deletion mutation failed on parent model {parent_id}')
        elif mutation_type is MutationType.DECOUPLE:
          self.debug_logger.debug(f'decouple mutation failed on parent model {parent_id}')
        elif mutation_type is MutationType.CHANNEL_CHANGE:
          self.debug_logger.debug(f'channel change mutation failed on model {parent_id}')
        elif mutation_type is MutationType.GREEN_MODEL_CHANGE:
          self.debug_logger.debug(f'green model change mutation failed on model {parent_id}')
        else:
          self.debug_logger.debug(f'group mutation failed on model {parent_id}')

        self.failed_mutation_info.append(self.current_mutation_info)
        failures += 1
        continue
      else: # successfully mutated tree
        try:
          check_channel_count(new_tree) # these should not fire...
        except AssertionError:
          self.debug_logger.debug(f"channel count check failed on model {model_id}")
          self.failed_mutation_info.append(self.current_mutation_info)
          failures += 1
          continue
        try:
          assert_no_nonlinear_adjacency(new_tree) # allow adjacent convs, just no adjacent relu / softmax
        except AssertionError:
          self.debug_logger.debug(f"check no adjacnet nonlinear failed on model {model_id}")
          self.failed_mutation_info.append(self.current_mutation_info)

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
          self.seen_models[new_tree]["model_ids"].append(int(model_id))
          seen_rejections += 1
          continue
      

@extclass(Mutator)
def green_model_change_mutation(self, tree):
  green_nodes = []
  green_model_id = None

  nodes = tree.preorder()

  for n in nodes:
    if type(n) is Input:
      if n.name == "Input(GreenExtractor)":
        green_nodes += [n]
        if green_model_id is None:  
          green_model_id = n.green_model_id
        else:
          assert green_model_id == n.green_model_id, "chroma DAG must use only one type of green model"
  
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
    shared = list(filter(lambda x: type(x.parent) is tuple and not type(x) in set((LogSub, AddExp, Input)), preorder_nodes))
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

  # remove chosen node from its partners' partner sets 
  # for each partner's partner set, replace the chosen node with the copies that are downstream of it
  if hasattr(chosen_node, "partner_set"):
    for copy_id, copied_node in enumerate(copies):
      assigned_parent = chosen_node_parents[copy_id]

      partners_upstream = find_partners_upstream(chosen_node.partner_set, assigned_parent)
      for partner in partners_upstream:
        if in_partner_set(partner.partner_set, chosen_node): # may be already removed by another copy that also feeds this partner
          remove_partner_from_set(partner.partner_set, chosen_node)
        partner.partner_set.add((copied_node, id(copied_node))) 

    update_partners_downstream_with_copies(chosen_node.partner_set, chosen_node, chosen_node, copies)

  return tree

def in_partner_set(partner_set, node):
  for p, pid in partner_set:
    if id(node) == pid:
      return True
  return False

def remove_partner_from_set(partner_set, node):
  for p, pid in partner_set:
    if id(node) == pid:
      partner_set.remove((p, pid))
      return
  assert False, "Could not find node to remove from partner set"

"""
given a set of partners and a starting node, traverses the 
branch up to the root node and returns all encountered partners
"""
def find_partners_upstream(partner_set, node):
  found_partners = set()
  while not node is None:
    if in_partner_set(partner_set, node):
      found_partners.add(node)
    node = node.parent
  return found_partners

"""
finds all downstream partners of node and replaces it in its
partners' partner sets with the copies of the node
"""
def update_partners_downstream_with_copies(partner_set, root, node, node_copies):
  if in_partner_set(partner_set, root):
    if in_partner_set(root.partner_set, node):
      remove_partner_from_set(root.partner_set, node)
    root.partner_set = root.partner_set.union( set([(n, id(n)) for n in node_copies]) )
  if root.num_children == 0:
    return 

  if root.num_children == 3:
    children = [root.child1, root.child2, root.child3]
  elif root.num_children == 2:
    children = [root.lchild, root.rchild]
  else:
    children = [root.child]

  for child in children:
    update_partners_downstream_with_copies(partner_set, root.child, node, node_copies)

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
    convs = list(filter(lambda x: type(x) in linear_ops, preorder_nodes))
    if len(convs) == 0:
      self.debug_logger.debug("cannot perform channel mutation on tree with no convs")
      assert False, "cannot perform channel mutation on tree with no convs"
    chosen_conv = random.sample(convs, 1)[0]

    for i, n in enumerate(preorder_nodes):
      if n is chosen_conv:
        chosen_conv_id = i
        break

  new_out_c = perturb_channel(chosen_conv.out_c)

  self.current_mutation_info.node_id = chosen_conv_id
  self.current_mutation_info.new_output_channels = new_out_c

  fixed = fix_channel_count_upwards(chosen_conv, new_out_c)
  if fixed:
    chosen_conv.out_c = new_out_c
    # if the mutated convolution's new output channels is not divisible by its groups,
    # set the groups to the factor of its in and out channels closest to its current groups
    in_c_factors = get_factors(chosen_conv.in_c)
    out_c_factors = get_factors(chosen_conv.out_c)
    factors = in_c_factors.intersection(out_c_factors)
    closest_factor = get_closest_factor(factors, chosen_conv.groups)
    chosen_conv.groups = closest_factor # if current groups is already a factor, this does nothing

    #print(f"changed channel count of {chosen_conv.dump()} to {new_out_c} full tree {tree.dump()}")
    return tree
  else:
    self.debug_logger.debug(f"Unable to perturb output channel of {chosen_conv} from {chosen_conv.out_c} to {new_out_c}")
    assert False, "Unable to perturb channel counts of conv"


"""
grouped conv mutation 
changes grouping factor of randomly chosen conv
"""
@extclass(Mutator)
def group_mutation(self, tree):
  preorder_nodes = tree.preorder()
  convs = list(filter(lambda x: type(x) in linear_ops, preorder_nodes)) 
  if len(convs) == 0:
    self.debug_logger.debug("cannot perform grouped conv mutation on tree with no convs")
    assert False, "cannot perform grouped conv mutation on tree with no convs"

  chosen_conv = random.sample(convs, 1)[0]
  for i, n in enumerate(preorder_nodes):
    if n is chosen_conv:
      chosen_conv_id = i
      break
  self.current_mutation_info.node_id = chosen_conv_id

  in_c_factors = get_factors(chosen_conv.in_c)
  out_c_factors = get_factors(chosen_conv.out_c)
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
adds partner node to node's set of partner nodes
"""
def add_partner(node, node_partner):
  if hasattr(node, "partner_set"):
    node.partner_set.add((node_partner, id(node_partner)))
  else:
    node.partner_set = set([(node_partner, id(node_partner))])

"""
given a dictionary of node ids to node references, return all of their partner nodes
"""  
def get_partner_nodes(id2node_map):
  partner_nodes = {}
  for nid, n in id2node_map.items():
    if hasattr(n, "partner_set"):
      for partner_node, partner_id in n.partner_set: 
        partner_nodes[partner_id] = partner_node      
  return partner_nodes


"""
given a dictionary of node ids to node references, return all of their partner nodes
that are still reachable through the DAG
"""
def get_reachable_partner_nodes(tree, id2node_map):
  partner_nodes = {}
  for nid, n in id2node_map.items():
    if hasattr(n, "partner_set"):
      for partner_node, partner_id in n.partner_set: 
        partner_nodes[partner_id] = partner_node  

  reachable = {}
  for nid, n in partner_nodes.items():
    if is_in_tree(tree, n):
      reachable[nid] = n
  return reachable

"""
given a dictionary of node ids to node references of nodes, 
returns the subset whos partners are not all eachable 
"""
def get_nodes_missing_partners(tree, id2node_map):
  nodes_missing_partners = {}
  for nid, n in id2node_map.items():
    for partner_node, partner_id in n.partner_set: 
      if not is_in_tree(tree, partner_node):
        nodes_missing_partners[nid] = n
      
  return nodes_missing_partners

"""
returns whether a node still exists in the tree
"""
def is_in_tree(tree, node):
  preorder_nodes = tree.preorder()
  for n in preorder_nodes:
    if id(n) == id(node):
      return True
  return False

"""
partners_of_deleted: dictionary of ids to nodes 
deleted: dictionary of ids to nodes

remove connections from nodes in partners_of_deleted to nodes in deleted
ONLY IF there is no longer a path from the partner to the deleted node 
because with DAGs, a partner can reference the same deleted node through 
multiple paths, and if only one path to the deleted node is removed, the 
deleted node still exists in the DAG
"""
def remove_deleted_partner_connections(tree, partners_of_deleted, deleted):
  for nid, n in partners_of_deleted.items():
    new_set = set()
    for partner, partner_id in n.partner_set: 
      # nodes may have been modified since insertion into partner set
      # so we need to reproduce the set by rehashing the nodes
      if not partner_id in deleted or is_in_tree(tree, partner): # don't think you need the first condiion
        new_set.add((partner, partner_id))
    n.partner_set = new_set 
    if len(n.partner_set) == 0:
      delattr(n, "partner_set")


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
  if parent.num_children == 3:
    if deletion_node is parent.child1:
      parent.child1 = new_child
    elif deletion_node is parent.child2:
      parent.child2 = new_child
    else:
      parent.child3 = new_child 
  elif parent.num_children == 2:
    if deletion_node is parent.lchild:
      parent.lchild = new_child
    else:
      parent.rchild = new_child 
  else:
    parent.child = new_child

  if type(new_child.parent) is tuple:
    new_child_parents = list(new_child.parent)
  else:
    new_child_parents = [new_child.parent]

  updated_new_child_parents = []
  for new_child_parent in new_child_parents:
    if not id(new_child_parent) == id(deletion_node):
      updated_new_child_parents += [new_child_parent] # keep all parents of new child that are not the deleted node

  updated_new_child_parents += [parent] # add the parent of the deleted node to the new child's list of parents
  if len(updated_new_child_parents) > 1:
    new_child.parent = tuple(updated_new_child_parents)
  else:
    new_child.parent = updated_new_child_parents[0]



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
  # if already_deleted is None:
  #   already_deleted = {}

  # if id(node) in already_deleted and not is_in_tree(tree, node):
  #   return tree
  if not is_in_tree(tree, node):
    return tree

  parent = node.parent
  deleted_nodes = {}

  self.debug_logger.debug(f"deleting {id(node)} {node.name}")

  deleted_nodes[id(node)] = node

  if isinstance(node, Binop):
    # if op is LogSub or AddExp always keep left child
    if type(node) is LogSub or type(node) is AddExp:
      keep_left = True
    else: # decide if you're going to keep left or right child
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
 
  # add newly deleted node to already_deleted
  # for deleted_id, deleted_n in deleted_nodes.items():
  #   already_deleted[deleted_id] = deleted_n

  remove_connections_to_deleted_nodes(tree, deleted_nodes)
  reachable_partners = get_reachable_partner_nodes(tree, deleted_nodes)
  nodes_missing_partners = get_nodes_missing_partners(tree, reachable_partners)
  # delete partners if we deleted a sandwich node
  #partners_of_deleted = get_partner_nodes(deleted_nodes)

  # filter out any partners that have already been deleted
  # filtered_partners_of_deleted = {}
  # for pid, p in partners_of_deleted.items():
  #   if not pid in already_deleted or is_in_tree(tree, p):
  #     filtered_partners_of_deleted[pid] = p

  # must remove partner connections that refer back to deleted_nodes - else infinite recursion deletion
  #remove_deleted_partner_connections(tree, filtered_partners_of_deleted, deleted_nodes)
  #remove_deleted_partner_connections(tree, reachable_partners, deleted_nodes)

  # for nid, n in filtered_partners_of_deleted.items():
  #   tree = self.delete_nodes(tree, n, already_deleted)
  for nid, n in nodes_missing_partners.items():
    tree = self.delete_nodes(tree, n)

  for n in illegal_parents:
    tree = self.delete_nodes(tree, n)

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

  self.debug_logger.debug(f"the chosen delete node {node} id {tree.get_preorder_id(node)}")
  self.current_mutation_info.node_id = tree.get_preorder_id(node)
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
# @extclass(Mutator)
# def get_insert_types(self, parent, child):
#   if type(parent) is tuple: 
#     assert isinstance(parent[0], Linear) and isinstance(parent[1], Linear) or\
#           isinstance(parent[0], NonLinear) and isinstance(parent[1], NonLinear) or\
#           isinstance(parent[0], Special) and isinstance(parent[1], Special)
#     parent = parent[0]
#   if isinstance(parent, Linear):
#     if isinstance(child, NonLinear):
#       flip = random.randint(0,10)
#       if flip < 7:
#         return [special_insert_ops]
#       else:
#         return [nonlinear_insert_ops, linear_insert_ops]
#     elif isinstance(child, Special):
#       return [nl_sp_insert_ops]
#     else:
#       self.debug_logger.debug("Linear parent cannot have Linear child")
#       assert False, "Linear parent cannot have Linear child"
#   elif isinstance(parent, NonLinear):
#     if isinstance(child, Linear):
#       flip = random.randint(0,10)
#       if flip < 7:
#         return [special_insert_ops]
#       else:
#         return [linear_insert_ops, nonlinear_insert_ops]
#     elif isinstance(child, Special):
#       return [l_sp_insert_ops]
#     else:
#       self.debug_logger.debug("NonLinear parent cannot have NonLinear child")
#       assert False, "NonLinear parent cannot have NonLinear child"
#   elif isinstance(parent, Special):
#     if isinstance(child, Linear):
#       return [nl_sp_insert_ops]
#     elif isinstance(child, NonLinear):
#       return [l_sp_insert_ops]
#     else:
#       return [all_insert_ops]

"""
connects insertion parent node to newly created child
insert_child: the node above which the insertion occured
insert_parent: the original parent (or one of the original parents) of insert_child
node: the node that was inserted between the child and the parent 
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
  elif insert_parent.num_children == 2:
    if insert_child is insert_parent.rchild:
      insert_parent.rchild = node
    else:
      insert_parent.lchild = node
  elif insert_parent.num_children == 1:
    insert_parent.child = node
  else:
    self.debug_logger.debug(f"Should not be inserting {node} after parent {insert_parent.dump()}")
    assert False, "Should not be inserting after an input node"

@extclass(Mutator)
def insert_binop(self, tree, input_set, insert_op, insert_parent, insert_child, make_copy=False):
  OpClass, use_child, use_right = insert_op
  if not hasattr(tree, "size"):
    tree.compute_size(set(), count_all_inputs=True)

  #### pick a subtree ####
  if issubclass(OpClass, BinopIII):
    # make subtree's output channels and spatial resolution match other child's output channels
    if use_child is None:
      if tree.size < self.args.min_subtree_size + 2:
        self.debug_logger.debug(f"Impossible to find subtree of size {self.args.min_subtree_size} or greater from tree of size {tree.size}")
        assert False, f"Impossible to find subtree of size {self.args.min_subtree_size} or greater from tree of size {tree.size}"
      subtree = self.pick_subtree(tree, input_set, insert_child, target_out_c=insert_child.out_c, resolution=spatial_resolution(insert_child), make_copy=make_copy)
    else:
      subtree = use_child
      # make channel count of insert_child agree with required use_child since we cannot change the required child
      fixed = fix_channel_count_downwards(insert_child, insert_parent, use_child.out_c)
      if not fixed:
        self.debug_logger.debug(f"Could not make channel counts of insert_child agree with required child")
        assert False, f"Could not make channel counts of insert_child agree with required child"
  else: # Op is BinopIJK - keep output channels of the chosen subtree unchanged but make spatial resolution match insert child's
    if tree.size < self.args.min_subtree_size + 2:
      self.debug_logger.debug(f"Impossible to find subtree of size {self.args.min_subtree_size} or greater from tree of size {tree.size}")
      assert False, f"Impossible to find subtree of size {self.args.min_subtree_size} or greater from tree of size {tree.size}"
    subtree = self.pick_subtree(tree, input_set, insert_child, resolution=spatial_resolution(insert_child), make_copy=make_copy)
  
  # check spatial resolutions of subtree and insert_child match 
  subtree_res = spatial_resolution(subtree)
  insert_child_res = spatial_resolution(insert_child)

  if subtree_res != insert_child_res:
    self.debug_logger.debug(f"resolution of chosen subtree for binary op does not match insert_child resolution")
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

  # parent is not none when inserting a required right child OR not making a copy of the chosen subtree
  if subtree.parent: 
    if type(subtree.parent) is tuple: 
      subtree.parent = subtree.parent + tuple([new_node])
    else:
      subtree.parent = (subtree.parent, new_node)
  else:
    subtree.parent = new_node

  if type(insert_child.parent) is tuple:
    parents = list(insert_child.parent)
    for p in insert_child.parent:
      if id(p) == id(insert_parent):
        parents.remove(p)
        parents += [new_node]
    insert_child.parent = tuple(parents)
  else:
    insert_child.parent = new_node  

  return new_node

"""
Inserts a unary op
resets insert_child's parent to point to newly created node
new node points to insert_child as its child
INSERT PARENT IS NOT MODIFIED TO POINT TO THE NEW NODE
"""
@extclass(Mutator)
def insert_unary_op(self, OpClass, insert_parent, insert_child):
  params = list(signature(OpClass).parameters.items())
  if len(params) >= 3 and len(params) <= 5: # conv or grouped/interleaved sum, possible params: child, out_c, name, groups, kwidth
    if issubclass(OpClass, UnopIIdiv):
      in_c_factors = get_factors(insert_child.out_c)
      # remove the input channels from the list of output channel options
      in_c_factors.remove(insert_child.out_c)
      assert len(in_c_factors) > 0, "Cannot insert {OpClass} without output channels to choose from"
      out_c = random.sample(in_c_factors, 1)[0]
    else:
      out_c = self.args.default_channels
    new_node = OpClass(insert_child, out_c)
  elif len(params) == 2:
    new_node = OpClass(insert_child)
  else:
    self.debug_logger.debug("Invalid number of parameters for Unary op")
    assert False, "Invalid number of parameters for Unary op"
  
  if type(insert_child.parent) is tuple:

    parents = dict([(id(p), p) for p in insert_child.parent])

    for p in insert_child.parent:
      if id(p) in parents:
        del parents[id(p)]
        parents[id(new_node)] = new_node

    insert_child.parent = tuple(parents.values())
  else:
    insert_child.parent = new_node  
  return new_node


"""
returns whether there is a downsample above the current node 
"""
def found_downsample_above(node):
  if isinstance(node, Downsample):
    return True
  else:
    if node.parent is None:
      return False
    if type(node.parent) is tuple:
      return any([found_downsample_above(p) for p in node.parent])
    else:
      return found_downsample_above(node.parent)


"""
checks if insertion location is ok for the given op, asserts False if not ok.
if OK, returns any necessary additional ops to keep nonlin / lin op adjacency
"""
def accept_insertion_loc(child, parent, insert_op):
  # Downsample cannot be parent of any Conv
  if insert_op is Downsample:
    found_node, found_level = find_type(child, Linear, 0, ignore_root=False)
    if any(found_node): 
      assert False, "rejecting insert location for downsample above a conv"
    found_node, found_level = find_type(child, Downsample, 0, ignore_root=False)
    if any(found_node): 
      assert False, "rejecting insert location for downsample above another Downsample"
    if found_downsample_above(parent):
      assert False, "rejecting insert location for downsample below another Downsample"
    if type(parent) in border_ops:
      assert False, "rejecting insert location for downsample directly below a border op"
    if type(parent) is Stack:
      assert False, "rejecting insert location for downsample directly below Stack"

  # reject two UnopIIdivs in a row (i.e. InterleavedSum / GroupedSum)
  if isinstance(insert_op, UnopIIdiv):
    if isinstance(parent, UnopIIdiv) or isinstance(child, UnopIIdiv):
      assert False, "rejecting insert location for UnopIIdiv next to another UnopIIdiv"

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
  insert_nodes = None
  if len(relu_or_softmax.intersection(childtypes)) == 0:
    insert_op, insert_child = choose_activation_under_mul(mul_node)
    insert_nodes, tree = self.insert(tree, insert_op, mul_node, insert_child, input_set)
  return insert_nodes, tree


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
    #insertion_child_options = find_closest_ancestor(op_node, set((Binop,)))[1:]

    # allow upsamples to be parent of certain binops: Subs and Adds --> requires fixing resolution of children
    # because we may subtract or add downsampled Green from/to downsampled Bayer
    # upsample cannot be inserted above a pre-existing downsample op or a Softmax, or Stack
    insertion_child_options = find_closest_ancestor(op_node, set((Downsample, Upsample, Softmax, Stack)))[1:]       
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
    if tries > self.args.partner_insert_loc_tries:
      self.debug_logger.debug(f"unable to find an insertion location for partner op of {op_node} with resolution {resolution}")
      assert False, f"unable to find an insertion location for partner op of {op_node} with resolution {resolution}"

  if OpClass is LogSub:
    # NOTE: we are passing a REFERENCE to LogSub's child -> creating a DAG
    insert_ops = [(partner_op_class, op_node.rchild, True)] # LogSub subtracts the right child so must add back
  else:
    insert_ops = [(partner_op_class, None, None)]

  insert_parent = insert_child.parent
  if type(insert_parent) is tuple:
    parent_types = [type(p) for p in insert_parent]
    if not LogSub in parent_types and not AddExp in parent_types:
      insert_parent = random.sample(insert_parent, 1)[0]
  return insert_ops, insert_parent, insert_child


"""
fixes the resolution of children downstream from newly inserted
Upsample when upsample is inserted above a Binop
prev_node is the parent that we traversed to arrive at the current node 
in the case where a node has multiple parents and not all of them need a 
Downsample inserted between it and the curr_node
"""
@extclass(Mutator)
def fix_res(self, tree, parent_upsample, input_set, curr_node, prev_node, path=None):
  if path is None:
    path = []
  
  path += [id(curr_node)]
  if isinstance(curr_node, Upsample) or isinstance(curr_node, Input):
    # insert downsample between curr_node and all curr_node's parents
    new_nodes, tree = self.insert(tree, [(Downsample, None, None)], prev_node, curr_node, input_set)
    inserted_downsample = new_nodes[0]
    add_partner(parent_upsample, inserted_downsample)
    add_partner(inserted_downsample, parent_upsample)

    # make sure if upstream paths are affected by this downsample insertion, and they need an upsample that we add it 
    # go up the DAG from the newly inserted Downsample until you either find an Upsample or you find a node with multiple
    # parents. If you find a node with multiple parents first, insert an Upsample right above the node with multiple parents 
    # between the node and all of its parents EXCEPT the parent from the path we just came down
    root = prev_node
    while not root is None:
      if type(root) is Upsample:
        break
      if type(root.parent) is tuple:
        root_parents = root.parent # must store local copy of parent tuple because insert operation can change the root.parent tuple
        for p in root_parents:
          if not id(p) in path:
            new_nodes, tree = self.insert(tree, [(Upsample, None, None)], p, root, input_set)
            inserted_upsample = new_nodes[0]
            add_partner(inserted_downsample, inserted_upsample)
            add_partner(inserted_upsample, inserted_downsample)
        return tree
      root = root.parent
    return tree 

  elif isinstance(curr_node, Unop):
    return self.fix_res(tree, parent_upsample, input_set, curr_node.child, curr_node, path)
  elif isinstance(curr_node, Binop):
    lres = spatial_resolution(curr_node.lchild)
    rres = spatial_resolution(curr_node.rchild)
    if lres != Resolution.DOWNSAMPLED:
      tree = self.fix_res(tree, parent_upsample, input_set, curr_node.lchild, curr_node, path)
    if rres != Resolution.DOWNSAMPLED:
      tree = self.fix_res(tree, parent_upsample, input_set, curr_node.rchild, curr_node, path)
    return tree

"""
inserts the partner op of the given op_node with class op_class
"""
@extclass(Mutator)
def insert_partner_op(self, tree, input_set, op_class, op_node):
  op_partner_class = sandwich_pairs[op_class]  
 
  try:
    insert_op, insert_parent, insert_child = self.choose_partner_op_loc(op_node, op_class, op_partner_class)
  except AssertionError:
    self.debug_logger.debug(f"failed to find insertion location for partner of {op_node}")
    assert False, f"failed to find insertion location for partner of {op_node}"
  
  insert_nodes, tree = self.insert(tree, insert_op, insert_parent, insert_child, input_set)
  
  partner_node = insert_nodes[0]
 
  if op_partner_class is Upsample:
    # fix spatial resolution of downstream children if Upsample inserted above binary op 
    if find_type_between(partner_node, op_node, Binop):
      inserted_upsample = partner_node
      self.fix_res(tree, inserted_upsample, input_set, partner_node.child, partner_node) 
   
  # assign pointers to partner nodes
  add_partner(op_node, partner_node)
  add_partner(partner_node, op_node)

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
def insert(self, tree, insert_ops, insert_parent, insert_child, input_set):
  cur_child = insert_child
  new_nodes = [] 

  for i, (OpClass, use_child, use_right) in enumerate(reversed(insert_ops)):
    cur_child.compute_input_output_channels()
    if issubclass(OpClass, Binop):
      params = list(signature(OpClass).parameters.items())
      assert len(params) == 3, f"Invalid number of parameters {len(params)} for Binary op"
      new_node = self.insert_binop(tree, input_set, (OpClass, use_child, use_right), insert_parent, cur_child)
     
    elif issubclass(OpClass, Unop):
      new_node = self.insert_unary_op(OpClass, insert_parent, cur_child)

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


"""
Tries to mutate tree by inserting random op(s) at a random location
If op is a binary op, picks a subtree to add as the other child
"""
@extclass(Mutator)
def insert_mutation(self, tree, input_set, insert_above_node_id=None, insert_op=None):
  preorder_nodes = tree.preorder()

  while True:
    # pick an op to insert
    if not insert_op:
      insert_op = random.sample(all_insert_ops, 1)[0]

    location_selection_failures = 0 

    while location_selection_failures < self.args.insert_location_tries:
      if not insert_above_node_id:
        insert_above_node_id = random.randint(1, len(preorder_nodes)-1)

      insert_child = preorder_nodes[insert_above_node_id]
      insert_parent = insert_child.parent

      if type(insert_parent) is tuple:
        parent_types = [type(p) for p in insert_parent]
        if not LogSub in parent_types and not AddExp in parent_types:
          insert_parent = random.sample(insert_parent, 1)[0]

      try:
        insert_ops = accept_insertion_loc(insert_child, insert_parent, insert_op)
        
      except AssertionError:
        insert_above_node_id = None
        location_selection_failures += 1
        if location_selection_failures >= self.args.insert_location_tries:
          insert_op = None 
  
        continue
     
      new_ops = [format_for_insert(o) for o in insert_ops]
     
      self.current_mutation_info.insert_ops = ";".join([new_op[0].__name__ for new_op in new_ops])
      self.current_mutation_info.node_id = insert_above_node_id

      try:
        new_nodes, tree = self.insert(tree, new_ops, insert_parent, insert_child, input_set)
      except AssertionError:
        assert False, "insertion mutation failed"
      
      """
      if we inserted a sandwich op, we must insert its partner node as well
      if we inserted Mul, and neither child is Softmax or Relu, add Softmax or Relu as a child
      """
      for (OpClass,_,_), new_node in zip(new_ops, new_nodes):
        if OpClass is Mul:
          _, tree = self.insert_activation_under_mul(tree, input_set, new_node)
        if OpClass in sandwich_ops:
          _, tree = self.insert_partner_op(tree, input_set, OpClass, new_node)

      return tree

  # while True:
  #   # randomly pick an op type to insert
  #   all_insert_ops
  #   # insert above the selected node
  #   if not insert_above_node_id:
  #     insert_above_node_id = random.randint(1, len(preorder_nodes)-1)
  #   insert_child = preorder_nodes[insert_above_node_id]
  #   insert_parent = insert_child.parent

  #   if type(insert_parent) is tuple:
  #     self.debug_logger.debug(f"insert parents {insert_parent}")
  #     parent_types = [type(p) for p in insert_parent]
  #     if not LogSub in parent_types and not AddExp in parent_types:
  #       insert_parent = random.sample(insert_parent, 1)[0]

  #   # pick op(s) to insert
  #   insert_types = self.get_insert_types(insert_parent, insert_child)
   
  #   while True:
  #     if not insert_op:
  #       new_ops = []
  #       for n in range(len(insert_types)):
  #         OpClass = random.sample(insert_types[n], 1)[0]
  #         new_ops += [OpClass]
  #     else:
  #       new_ops = [insert_op]
  #     # allow at most one sandwich op to be inserted
  #     if sum(map(lambda x : x in sandwich_ops, new_ops)) <= 1: 
  #       break

  #   if self.accept_insertion_choice(insert_child, insert_parent, new_ops):
  #     self.debug_logger.debug(f"accepted insertion choice insert child: {insert_child.dump()} new ops {new_ops}")
  #     break

  # new_ops = [format_for_insert(o) for o in new_ops]

  # self.current_mutation_info.insert_ops = ";".join([new_op[0].__name__ for new_op in new_ops])
  # self.current_mutation_info.node_id = insert_above_node_id

  # try:
  #   new_nodes, tree = self.insert(tree, new_ops, insert_parent, insert_child, input_set)
  # except AssertionError:
  #   if Downsample in new_ops:
  #     self.debug_logger.debug(f"failed to insert downsample into tree:\n{tree.dump()}\nwith insert_child\n{insert_child.dump()}")
  #   assert False, "insertion mutation failed"
    
  # # if we inserted a sandwich op, we must insert its partner node as well
  # # if we inserted Mul, and neither child is Softmax or Relu, add Softmax or Relu as a child
  # """
  #  NOTE: not ideal that this pruning rule is being muddled in as an insertion rule
  # """
  # for (OpClass,_,_), new_node in zip(new_ops, new_nodes):
  #   if OpClass is Mul:
  #     _, tree = self.insert_activation_under_mul(tree, input_set, new_node)
  #   if OpClass in sandwich_ops:
  #     _, tree = self.insert_partner_op(tree, input_set, OpClass, new_node)
  # return tree


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
  elif tree.num_children == 2:
    lchild_pool = splits_sandwich_ops_helper(tree.lchild, partner_pool)
    partner_pool = splits_sandwich_ops_helper(tree.rchild, lchild_pool)
  elif tree.num_children == 1:
    partner_pool = splits_sandwich_ops_helper(tree.child, partner_pool)
  return partner_pool


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
def allow_subtree(self, root, input_set, insert_child, target_out_c=None, resolution=None, make_copy=False):
  if id(insert_child) == id(root):
    self.debug_logger.debug(f"chosen subtree cannot be the same tree as other child of binop")
    assert False, f"chosen subtree cannot be the same tree as other child of binop "

  subtree_resolution = spatial_resolution(root)
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

  # reject subtrees that split up sandwich nodes LogSub and AddExp
  # NOTE : ALLOWING SUBTREES THAT SPLIT DOWN/UPSAMPLE IS CONDITIONED ON SPATIAL RESOLUTIONS MATCHING
  splits = splits_sandwich_ops(root)
  assert (not splits), f"rejecting subtree: splits sandwich_ops"
  
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
"""
@extclass(Mutator)
def pick_subtree(self, root, input_set, insert_child, target_out_c=None, resolution=None, root_id=None, make_copy=False):
  preorder = root.preorder()
  failures = 0
  while True:
    if root_id and failures == 0:
      subtree_id = root_id
    else:
      subtree_id = random.randint(1, len(preorder)-1)
    subtree = preorder[subtree_id]
    subtree.compute_input_output_channels()
    try: 
      chosen_subtree = self.allow_subtree(subtree, input_set, insert_child, target_out_c, resolution, make_copy)
    except AssertionError:
      failures += 1
      self.debug_logger.debug("selected subtree is invalid")
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
  if len(tree.preorder()) > self.args.max_nodes:
    self.debug_logger.debug(f"rejecting tree with size {len(tree.preorder())} larger than max tree size")
    return False

  if spatial_resolution(tree) == Resolution.INVALID:
    self.debug_logger.debug("rejecting invalid spatial resolution")
    return False
    
  # reject DAGs with receptive fields larger than threshold
  footprint = tree.compute_footprint()
  if footprint > self.args.max_footprint:
    self.debug_logger.debug(f"rejecting DAG with footprint {footprint}")
    return False

  # reject DAGs with channel counts larger than threshold
  max_channels = get_max_channels(tree)
  if max_channels > self.args.max_channels:
    self.debug_logger.debug(f"rejecting tree with max channels {max_channels}")
    return False

  tree_type = type(tree) 
  if tree_type is InterleavedSum or tree_type is GroupedSum: 
    # don't sum reduce over one channel
    if tree.child.out_c == 1:
      self.debug_logger.debug("rejecting sum over one channel")
      return False

  # don't allow subtraction or addition of same trees
  if tree_type is Sub or tree_type is Add:
    if tree.lchild.is_same_as_wrapper(tree.rchild):
      self.debug_logger.debug("Rejecting sub or add same tree")
      return False
    # don't allow addition or subtraction of linear subtrees
    if is_linear(tree):
      self.debug_logger.debug("Rejecting sub or add linear trees")
      return False
  
  # don't allow LogSub of same trees
  if tree_type is LogSub:
    if tree.lchild.is_same_as_wrapper(tree.rchild):
      self.debug_logger.debug("rejecting LogSub same tree")
      return False

  # output channels of Conv1D must be divisible by 2
  if tree_type is Conv1D:
    if tree.out_c % 2 != 0:
      self.debug_logger.debug("rejecting Conv1D with odd output channels")
      return False
      
  # don't allow stacking of same trees
  # if tree_type is Stack:
  #   if tree.lchild.is_same_as(tree.rchild):
  #     self.debug_logger.debug("rejecting stacking same children")
  #     return False

  # don't allow softmax over a single channel
  if tree_type is Softmax and tree.out_c == 1:
    return False

  # don't allow two upsamples / downsamples or two softmaxes in same branch
  if tree_type is Upsample or tree_type is Softmax:
    ancestors_from_binop_to_node = find_closest_ancestor(tree, set((Binop,)))
    instances = sum([1 for n in ancestors_from_binop_to_node if type(n) is tree_type])
    if instances > 1:
      self.debug_logger.debug("Rejecting sub or add linear trees")
      return False

  # don't allow adjacent LogSub / AddExp or adjacent Downsample / Upsample
  if tree_type is LogSub:
    if type(tree.parent) is AddExp:
      self.debug_logger.debug("rejecting adjacent LogSub / AddExp")
      return False
  if tree_type is Downsample:
    if type(tree.parent) is Upsample:
      self.debug_logger.debug(f"rejecting adjacent Down / Up in {tree.parent.dump()}")
      return False

    # downsample cannot be parent of any conv
    found_node, found_level = find_type(tree.child, Linear, 0, ignore_root=False)
    if any(found_node):
      self.debug_logger.debug("rejecting downsample with a Conv child")
      return False

    # downsample cannot be parent of an upsample
    found_node, found_level = find_type(tree.child, Upsample, 0, ignore_root=False)
    if any(found_node):
      self.debug_logger.debug("rejecting downsample with an Upsample child")
      return False 

  # Mul must have either Softmax or Relu as one of its children
  # and do not allow two Mul in a row
  if tree_type is Mul:
    if type(tree.rchild) is Mul or type(tree.lchild) is Mul:
      self.debug_logger.debug("rejecting two consecutive Mul")
      return False
    relu_or_softmax = set((Relu, Softmax))
    childtypes = set((type(tree.lchild), type(tree.rchild)))
    if len(relu_or_softmax.intersection(childtypes)) == 0:
      self.debug_logger.debug("rejecting Mul without ReLU or Softmax child")
      return False

  if tree.num_children == 0:
    return True
  elif tree.num_children == 2:
    return self.accept_tree(tree.lchild) and self.accept_tree(tree.rchild)
  elif tree.num_children == 1:
    return self.accept_tree(tree.child)
  elif tree.num_children == 3:
    return self.accept_tree(tree.child1) and self.accept_tree(tree.child2) and self.accept_tree(tree.child3)
 
