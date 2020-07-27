# while mutating
# for pool in cost_tiers
#     for model in pool
#     new_model = mutate(model)   
#       cost = cost(new_model)
#     new_pool = get_cost_tier(cost)
#     new_pool.add(cost, new_model)
# for pool in cost_tiers:
#   pool = keep_topk(pool)
#
import logging
import sys
import time
import os
import argparse
import glob
import copy
import random
import numpy as np

import torch
from tree import has_loop
from cost import ModelEvaluator, CostTiers
from demosaic_ast import structural_hash
from mutate import Mutator
import util
from model_database import ModelDatabase 
import meta_model
from model_lib import multires_green_model, multires_green_model2, multires_green_model3

def zerochannels(tree):
  if tree.out_c == 0:
    return True
  if tree.num_children == 2:
    return zerochannels(tree.lchild) or zerochannels(tree.rchild)
  if tree.num_children == 1:
    return zerochannels(tree.child)
  return False

class Searcher():
  def __init__(self, args, search_logger, debug_logger):
    self.args = args  
    self.search_logger = search_logger
    self.debug_logger = debug_logger
    self.mutator = Mutator(args, debug_logger)
    self.evaluator = ModelEvaluator(args)
    self.model_manager = util.ModelManager(args.model_path)
    self.model_database = ModelDatabase(args.model_database_dir)
    
  # searches over program mutations within tiers of computational cost
  def search(self, compute_cost_tiers, tier_size):
    # initialize cost tiers
    cost_tiers = CostTiers(compute_cost_tiers)

    seed_model, seed_ast = util.load_model_from_file(self.args.seed_model_file, 0, cpu=True)
    seed_model_dir = self.model_manager.model_dir(self.model_manager.SEED_ID)
    util.create_dir(seed_model_dir)
    seed_ast.compute_input_output_channels()
    self.model_manager.save_model([seed_model], seed_ast, seed_model_dir)

    seed_ast.compute_input_output_channels()
    print(seed_ast.dump())
    # give random costs to initial tree
    compute_cost = self.evaluator.compute_cost(seed_ast)
    model_accuracy = self.args.seed_model_accuracy

    cost_tiers.add(self.model_manager.SEED_ID, compute_cost, model_accuracy)
    
    self.model_database.add(self.model_manager.SEED_ID,\
                        [self.model_manager.SEED_ID, 
                        structural_hash(seed_ast), 
                        1, 
                        0, 
                        [model_accuracy, -1, -1], 
                        compute_cost, 
                        -1,
                        0,
                        0,
                        0,
                        0,
                        0])

    # CHANGE TO NOT BE FIXED - SHOULD BE INFERED FROM TASK
    model_inputs = set(("Input(Bayer)",))
    t0 = time.time()
    for generation in range(self.args.generations):
      new_cost_tiers = copy.deepcopy(cost_tiers) 
      avg_iter_time = 0 
      for tier in cost_tiers.tiers:
        for model_id, costs in tier.items():
          itert0 = time.time()
          best_model_version = self.model_database.get_best_version_id(model_id)
          model, model_ast = self.model_manager.load_model(model_id, best_model_version)
          
          if has_loop(model_ast):
            print("tree has loop!!!")
            print(model_ast.dump())
            exit()

          new_model_id = self.model_manager.get_next_model_id()
          new_model_ast, shash, mutation_stats = self.mutator.mutate(new_model_id, model_ast, model_inputs)
          
          if new_model_ast is None: # mutation failed 
            self.model_database.increment_killed_mutations(model_id)
            continue

          new_models = [new_model_ast.ast_to_model() for i in range(args.model_initializations)]
          for m in new_models:
            m._initialize_parameters()

          new_model_dir = self.model_manager.model_dir(new_model_id)
          util.create_dir(new_model_dir)

          perf_costs = [random.uniform(0.0009, 0.0018) for i in range(args.model_initializations)]
          compute_cost = self.evaluator.compute_cost(new_model_ast)

          min_perf_cost = min(perf_costs)
          best_new_model_version = perf_costs.index(min_perf_cost)

          self.model_manager.save_model(new_models, new_model_ast, new_model_dir)
          new_cost_tiers.add(new_model_id, compute_cost, min_perf_cost)

          self.model_database.add(new_model_id,\
                        [new_model_id, 
                        shash,
                        1, # model occurrences 
                        best_new_model_version,
                        perf_costs,
                        compute_cost,
                        model_id,
                        0, # killed mutations from this model 
                        mutation_stats.failures, 
                        mutation_stats.prune_rejections, 
                        mutation_stats.structural_rejections,
                        mutation_stats.seen_rejections])

          itert1 = time.time()
          avg_iter_time += (itert1 - itert0) / sum([len(tier.items()) for tier in cost_tiers.tiers])

      new_cost_tiers.keep_topk(tier_size)
      cost_tiers = new_cost_tiers

      if generation % self.args.database_save_freq == 0:
        self.update_model_occurences()
        self.model_database.save()

      t1 = time.time()
      print(f"generation {generation} time {t1-t0} seen models {len(self.mutator.seen_models)}")
      t0 = time.time()
      print(f"avg iter time {avg_iter_time}")
    
    self.update_model_occurences()
    self.model_database.save()

    for m in self.mutator.seen_models:
      print(self.mutator.seen_models[m])
    return cost_tiers

  """
  occasionally consult mutator to update model occurences 
  """
  def update_model_occurences(self):
    for model in self.mutator.seen_models:
      model_ids = self.mutator.seen_models[model]
      occurences = len(model_ids)
      for model_id in model_ids:
        self.model_database.update_occurence_count(model_id, occurences)


def parse_cost_tiers(s):
  ranges = s.split(' ')
  ranges = [[int(x) for x in r.split(',')] for r in ranges]
  print(ranges)
  return ranges

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--seed_model_accuracy', type=float, default=0.0013, help='accuracy of seed model')
  parser.add_argument('--default_channels', type=int, default=16, help='num of output channels for conv layers')
  parser.add_argument('--max_nodes', type=int, default=30, help='max number of nodes in a tree')
  parser.add_argument('--min_subtree_size', type=int, default=2, help='minimum size of subtree in insertion')
  parser.add_argument('--max_subtree_size', type=int, default=11, help='maximum size of subtree in insertion')
  parser.add_argument('--structural_sim_reject', type=float, default=0.66, help='rejection probability threshold for structurally similar trees')
  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--model_database_dir', type=str, default='model_database', help='path to save model statistics')
  parser.add_argument('--database_save_freq', type=int, default=5, help='model database save frequency')
  parser.add_argument('--save', type=str, default='SEARCH_MODELS', help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--generations', type=int, default=20, help='model search generations')
  parser.add_argument('--seed_model_file', type=str, help='')
  parser.add_argument('--cost_tiers', type=str, help='list of tuples of cost tier ranges')
  parser.add_argument('--tier_size', type=int, default=10, help='how many models to keep per tier')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--mutation_failure_threshold', type=int, default=1000, help='max number of tries to mutate a tree')
  args = parser.parse_args()
  random.seed(args.seed)
  np.random.seed(args.seed)

  args.cost_tiers = parse_cost_tiers(args.cost_tiers)
  util.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  args.model_path = os.path.join(args.save, args.model_path)
  util.create_dir(args.model_path)
  args.model_database_dir = os.path.join(args.save, args.model_database_dir)
  util.create_dir(args.model_database_dir)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  search_logger = util.create_logger('search_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'search_log'))
  debug_logger = util.create_logger('debug_logger', logging.DEBUG, log_format, \
                                os.path.join(args.save, 'debug_log'))

  search_logger.info("args = %s", args)

  searcher = Searcher(args, search_logger, debug_logger)
  searcher.search(args.cost_tiers, args.tier_size)
