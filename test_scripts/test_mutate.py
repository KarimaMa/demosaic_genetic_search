import time
from test_models import *
from mutate import Mutator, accept_tree
from demosaic_ast import *
from type_check import check_channel_count, check_linear_types
import copy
import meta_model
from model_lib import *
import random
from graph_viz import vis_ast
import logging
import os
import util

"""
green_model = build_green_model()
full_model, inputs = build_full_model(green_model)
print("full model inputs")
print(inputs)
derived_inputs = full_model.get_inputs()
treestr = full_model.dump()
print(treestr)

check_linear_types(full_model)
allowed_inputs = inputs

print("--------------- MUTATING TREE ------------------")
multires_mut = False
delete_mut = delete_mutation(full_model)
print("model after deletion")
print(delete_mut.dump())
mut1 = insert_mutation(delete_mut, allowed_inputs, insert_op=Downsample)
print("model after insertion")
print(mut1.dump())

if multires_mut:
  mut1 = insert_mutation(full_model, allowed_inputs, insert_above_node_id=19, insert_op=Stack)
  print("model after subtree insertion")
  print(mut1.dump())
  preorder = mut1.preorder()
  mut2 = insert_mutation(mut1, allowed_inputs, insert_above_node_id=26, insert_op=Downsample)
  check_linear_types(mut2)
  print("the new model after downsample mutation")
  print(mut2.dump())
"""

def find_multires_green_from_default_green(mutator):
  full_model = meta_model.MetaModel()
  full_model.build_default_model() 
  print("------ testing model equality ---------")
  multires_green = multires_green_model()
  print("multires model {}".format(multires_green.dump()))
  print("size {}".format(len(multires_green.preorder())))
  multires_green2 = multires_green_model2()
  print("multires model 2 {}".format(multires_green2.dump()))
  print("size {}".format(len(multires_green2.preorder())))
  multires_green3 = multires_green_model3()
  print("multires model 3 {}".format(multires_green3.dump()))
  print("size {}".format(len(multires_green3.preorder())))
  green = full_model.green
    
  print("--------------- MUTATING META MODEL ------------------")
  allowed_inputs = full_model.green_inputs
  failed_mutations = 0
  mutations = 0
  rejects = 0

  t0 = time.time()
  seen_models = {}
  seen_structures = set()

  while not green.is_same_as(multires_green) and not green.is_same_as(multires_green2) and not green.is_same_as(multires_green3):
    try:
      while True:
        green_copy = copy.deepcopy(full_model.green)
        #mut1 = mutator.insert_mutation(green_copy, allowed_inputs, insert_above_node_id=10, insert_op=Stack)
        mut1 = mutator.insert_mutation(green_copy, allowed_inputs)
        if accept_tree(mut1):
          break 
        else:
          rejects += 1
    except AssertionError:
      failed_mutations += 1
      continue
    else:
      try:
        while True:
          mut1copy = copy.deepcopy(mut1)
          green = mutator.insert_mutation(mut1copy, allowed_inputs)
          #green = mutator.insert_mutation(mut1copy, allowed_inputs, insert_above_node_id=18, insert_op=Downsample)
          # print(green.dump())
          # print("------------")
          # print(multires_green.dump())
          # print(green.is_same_as(multires_green))
          # print("=============")
          if accept_tree(green):
            break
      except AssertionError:
        failed_mutations += 1
        continue
      else:
        # reject with p % chance similar trees
        h = structural_hash(green)
        if h in seen_structures:
          if random.randint(0,2) != 2:
            continue
        else:
          seen_structures.add(h)

        check_channel_count(green)
        check_linear_types(green)
        mutations += 1

    if green in seen_models:
      seen_models[green] += 1
    else:
      seen_models[green] = 1
    #print("final model {}".format(green.dump()))
    if mutations % 500 == 0:
      t1 = time.time()
      print("time for 500: {}".format(t1-t0))
      t0 = time.time()
      print("MUTATIONS {} rejects {} unique models {}".format(mutations, rejects, len(seen_models)))

  print("took {} mutations to make green model mutate to multires".format(mutations))
  print("had {} failed mutations".format(failed_mutations))
  print("mutated model {}".format(green.dump()))
  print("number of unique models {}".format(len(seen_models)))
  
  import operator
  sorted_models = sorted(seen_models.items(), key=operator.itemgetter(1))
  sorted_models.reverse()

  for i in range(min(100, len(sorted_models))):
    print("model occurred {} times".format(sorted_models[i][1]))
    print(sorted_models[i][0].dump())
  print("----------")


def test_deletion_and_insertion():
  full_model = meta_model.MetaModel()
  full_model.build_default_model() 
  green = full_model.green
  
  def pp(model):
    if model.parent:
      print("{} parents {}".format(model.name, model.parent.name))
    else:
      print("{}".format(model.name))
    if model.num_children == 2:
      pp(model.lchild)
      pp(model.rchild)
    if model.num_children == 1:
      pp(model.child)
  print(green.dump())
  pp(green)

  print("--------------- MUTATING GREEN MODEL ------------------")
  allowed_inputs = full_model.green_inputs
  failed_mutations = 0
  mutations = 0
  rejects = 0
  logsubs = 0

  t0 = time.time()
  seen_models = {}
  seen_structures = set()

  while mutations < 8000:
    try:
      while True:
        green_copy = copy.deepcopy(full_model.green)
        mut1 = insert_mutation(green_copy, allowed_inputs)
        if accept_tree(mut1):
          break 
        else:
          rejects += 1
    except AssertionError:
      failed_mutations += 1
      continue
    else:
      try:
        while True:
          mut1copy = copy.deepcopy(mut1)
          green = delete_mutation(mut1copy)
          if accept_tree(green):
            break
          else:
            rejects += 1
      except AssertionError:
        failed_mutations += 1
        continue
      else:
        # reject with p % chance similar trees
        h = structural_hash(green)
        if h in seen_structures:
          if random.randint(0,2) != 2:
            continue
        else:
          seen_structures.add(h)

        check_channel_count(green)
        check_linear_types(green)
        mutations += 1

    if green in seen_models:
      seen_models[green] += 1
    else:
      seen_models[green] = 1
    
    pn = green.preorder()
    if any([type(n) is LogSub for n in pn]):
      logsubs += 1

    if mutations % 500 == 0:
      t1 = time.time()
      print("time for 500: {}".format(t1-t0))
      t0 = time.time()
      print("MUTATIONS {} logsubs {} rejects {}  unique models {}".format(mutations, logsubs, rejects,len(seen_models)))

  print("mutations {} failed mutations {}".format(mutations, failed_mutations))
  print("number of unique models {}".format(len(seen_models)))
  
  import operator
  sorted_models = sorted(seen_models.items(), key=operator.itemgetter(1))
  sorted_models.reverse()

  for i in range(min(200, len(sorted_models))):
    print("model occurred {} times".format(sorted_models[i][1]))
    print(sorted_models[i][0].dump())
  print("----------")


def find_default_green_from_multires(multires, mutator):
  full_model = meta_model.MetaModel()
  full_model.build_default_model() 
  green = full_model.green

  graph = vis_ast(green, 'green_model')
  green.compute_size(set(), count_all_inputs=True)
  print(green.size)
  graph.render()

  graph = vis_ast(multires, 'multires')
  multires.compute_size(set(), count_all_inputs=True)
  print(multires.size)
  graph.render()

  print("--------------- MUTATING GREEN MODEL ------------------")

  allowed_inputs = full_model.green_inputs
  failed_mutations = 0
  mutations = 0
  rejects = 0

  t0 = time.time()
  seen_models = {}
  seen_structures = set()
  mut2 = copy.deepcopy(multires)

  while not mut2.is_same_mod_channels(green): 
    try:
      while True:
        multires_copy = copy.deepcopy(multires)
        mut1 = mutator.delete_mutation(multires_copy)
        if accept_tree(mut1):
          break 
        else:
          rejects += 1
    except AssertionError:
      failed_mutations += 1
      continue
    else:
      try:
        while True:
          mut1copy = copy.deepcopy(mut1)
          mut2 = mutator.delete_mutation(mut1copy)
          if accept_tree(mut2):
            break
          else:
            rejects += 1
      except AssertionError:
        failed_mutations += 1
        continue
      else:
        # reject with p % chance similar trees
        h = structural_hash(green)
        if h in seen_structures:
          if random.randint(0,2) != 2:
            continue
        else:
          seen_structures.add(h)

        check_channel_count(green)
        check_linear_types(green)
        mutations += 1

    if mut2 in seen_models:
      seen_models[mut2] += 1
    else:
      seen_models[mut2] = 1

    print("model after deletion {}".format(mut2.dump()))
    
    if mutations % 500 == 0:
      t1 = time.time()
      print("time for 500: {}".format(t1-t0))
      t0 = time.time()
      print("MUTATIONS {} rejects {} unique models {}".format(mutations, rejects,len(seen_models)))

  print("took {} mutations to make multires model mutate to green".format(mutations))
  print("had {} failed mutations".format(failed_mutations))
  print("number of unique models {}".format(len(seen_models)))
  print("mutated model {}".format(mut2.dump()))
  print("----------")
  
  import operator
  sorted_models = sorted(seen_models.items(), key=operator.itemgetter(1))
  sorted_models.reverse()

  for i in range(min(200, len(sorted_models))):
    print("model occurred {} times".format(sorted_models[i][1]))
    print(sorted_models[i][0].dump())
  print("----------")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--save', type=str, default='test_mutate_log', help='where to store test log')
  parser.add_argument('--default_channels', type=int, default=16, help='num of output channels for conv layers')
  parser.add_argument('--max_nodes', type=int, default=30, help='max number of nodes in a tree')
  parser.add_argument('--min_subtree_size', type=int, default=2, help='minimum size of subtree in insertion')
  parser.add_argument('--max_subtree_size', type=int, default=11, help='maximum size of subtree in insertion')
  parser.add_argument('--structural_sim_reject', type=float, default=0.66, help='rejection probability threshold for structurally similar trees')
  args = parser.parse_args()

  util.create_dir(args.save)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
  debug_logger = util.create_logger('debug_logger', logging.DEBUG, log_format, \
                                os.path.join(args.save, 'debug_log'))

  mutator = Mutator(args, debug_logger)

  find_multires_green_from_default_green(mutator)
  #test_deletion_and_insertion()
  # multires = multires_green_model()
  # find_default_green_from_multires(multires, mutator)
 
