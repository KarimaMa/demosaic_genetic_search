
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
import sys
sys.path.append(sys.path[0].split("/")[0])
from tree import has_loop
from cost import ModelEvaluator, CostTiers
from demosaic_ast import structural_hash, Downsample
from mutate import Mutator, has_downsample
import util
from model_database import ModelDatabase 
import meta_model
from model_lib import multires_green_model, multires_green_model2, multires_green_model3
from util import PerfStatTracker
from type_check import check_channel_count, shrink_channels



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
    self.model_database = ModelDatabase(args.model_database_dir, args.model_initializations)
    
  # searches over program mutations within tiers of computational cost
  def search(self, compute_cost_tiers, tier_size):
    performance_tracker = PerfStatTracker()

    cost_tiers = CostTiers(compute_cost_tiers)

    seed_model, seed_ast = util.load_model_from_file(self.args.seed_model_file, 0, 'cpu')
    seed_model_dir = self.model_manager.model_dir(self.model_manager.SEED_ID)
    util.create_dir(seed_model_dir)
    seed_ast.compute_input_output_channels()
    self.model_manager.save_model([seed_model], seed_ast, seed_model_dir)

    seed_ast.compute_input_output_channels()
    print(seed_ast.dump())
    # give madeup accuracy cost to initial tree - just testing that system works
    compute_cost = self.evaluator.compute_cost(seed_ast)
    model_accuracy = 0.0008

    cost_tiers.add(self.model_manager.SEED_ID, compute_cost, model_accuracy)
    
    self.model_database.add(self.model_manager.SEED_ID,\
            {'model_id': self.model_manager.SEED_ID,
             'hash': hash(seed_ast),
             'structural_hash': structural_hash(seed_ast),
             'occurrences': 1,
             'best_model_initialization': 0,
             'model_losses': [model_accuracy],
             'model_compute_cost': compute_cost,
             'parent_id': -1,
             'killed_children': 0,
             'failed_mutations': 0,
             'prune_rejections': 0,
             'structural_rejections': 0,
             'seen_rejections': 0})

    # CHANGE TO NOT BE FIXED - SHOULD BE INFERED FROM TASK
    model_inputs = set(("Input(Bayer)",))
    t0 = time.time()

    for generation in range(self.args.generations):
      generational_tier_sizes = [len(tier.items()) for tier in cost_tiers.tiers]
      print("---------------")
      printstr = "tier sizes: "
      for t in generational_tier_sizes:
        printstr += f"{t} "
      print(printstr)

      produced_models = 0

      generational_killed = 0
      generational_pruned_mutations = 0
      generational_failed_mutations = 0
      generational_seen_rejections = 0
      generational_structural_rejections = 0

      new_cost_tiers = copy.deepcopy(cost_tiers) 
      avg_iter_time = 0 

      while True:
        held_models = 0
        for tier in cost_tiers.tiers:
          held_models += len(tier)
        if produced_models > held_models * 0.3:
          break
        print(f"produced_models {produced_models} needed: {held_models * 0.3}")
        for tid, tier in enumerate(cost_tiers.tiers):
          for model_id, costs in tier.items():
            itert0 = time.time()
            best_model_initialization = self.model_database.get_best_initialization_id(model_id)
            model, model_ast = self.model_manager.load_model(model_id, best_model_initialization)
            
            if has_loop(model_ast):
              self.debug_logger.info("Tree has loop!!")
              self.debug_logger.info(model_ast.dump())
              exit()

            print(f"\n----- parent model before mutation -----")
            print(model_ast.dump())

            new_model_id = self.model_manager.get_next_model_id()
            mut0 = time.time()
            new_model_ast, shash, mutation_stats = self.mutator.mutate(new_model_id, model_ast, model_inputs)
            mut1 = time.time()
            performance_tracker.update("mutation", mut1-mut0)

            if not new_model_ast is None:
              print("child model")
              print(new_model_ast.dump())

            if new_model_ast is None: # mutation failed 
              self.model_database.increment_killed_mutations(model_id)
              generational_pruned_mutations += mutation_stats.prune_rejections
              generational_failed_mutations += mutation_stats.failures
              generational_structural_rejections += mutation_stats.structural_rejections
              generational_seen_rejections += mutation_stats.seen_rejections 
              generational_killed += 1
              continue
            
            compute_cost = self.evaluator.compute_cost(new_model_ast)

            if compute_cost > new_cost_tiers.max_cost:
              self.debug_logger.info(f"model with cost {compute_cost} is too computationally expensive")
              continue

            # if has_downsample(new_model_ast):
            #   print(new_model_ast.dump())

            new_models = [new_model_ast.ast_to_model() for i in range(args.model_initializations)]
            produced_models += 1

            for m in new_models:
              m._initialize_parameters()

            new_model_dir = self.model_manager.model_dir(new_model_id)
            util.create_dir(new_model_dir)

            perf_costs = [random.uniform(0.00045, 0.0015) for i in range(args.model_initializations)]
            min_perf_cost = min(perf_costs)
            best_new_model_initialization = perf_costs.index(min_perf_cost)

            self.model_manager.save_model(new_models, new_model_ast, new_model_dir)

            new_cost_tiers.add(new_model_id, compute_cost, min_perf_cost)

            self.model_database.add(new_model_id,\
              {'model_id': new_model_id,
               'hash': hash(new_model_ast),
               'structural_hash': shash,
               'occurrences': 1,
               'best_model_initialization': best_new_model_initialization,
               'model_losses': perf_costs,
               'model_compute_cost': compute_cost,
               'parent_id': model_id,
               'killed_children': 0,
               'failed_mutations': mutation_stats.failures,
               'prune_rejections': mutation_stats.prune_rejections,
               'structural_rejections': mutation_stats.structural_rejections,
               'seen_rejections': mutation_stats.seen_rejections})

            itert1 = time.time()
            avg_iter_time += (itert1 - itert0) / sum([len(tier.items()) for tier in cost_tiers.tiers])

      new_cost_tiers.keep_topk(tier_size)
      cost_tiers = new_cost_tiers

      if generation % self.args.database_save_freq == 0:
        self.update_model_occurences()
        self.model_database.save()

      t1 = time.time()
      performance_tracker.update("total", t1-t0)
      print(f"generation {generation} time {t1-t0} seen models {len(self.mutator.seen_models)}")
      t0 = time.time()
      print(f"time spent in total {performance_tracker.function_time_sums['total']}")
      print(f"time spent in mutation {performance_tracker.function_time_sums['mutation']}")
      print(f"proportion spent in mutation {performance_tracker.function_time_sums['mutation']/performance_tracker.function_time_sums['total']}")
      print(f"number of killed mutations {generational_killed} caused by:")
      print(f"seen_rejections {generational_seen_rejections} prune_rejections {generational_pruned_mutations}" 
            + f" structural_rejections {generational_structural_rejections} failed_mutations {generational_failed_mutations}")

    self.update_model_occurences()
    self.model_database.save()

    return cost_tiers

  """
  occasionally consult mutator to update model occurences 
  """
  def update_model_occurences(self):
    for model, value in self.mutator.seen_models.items():
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
  parser.add_argument('--default_channels', type=int, default=16, help='num of output channels for conv layers')
  parser.add_argument('--max_nodes', type=int, default=33, help='max number of nodes in a tree')
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
  parser.add_argument('--tier_size', type=int, default=20, help='how many models to keep per tier')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--mutation_failure_threshold', type=int, default=500, help='max number of tries to mutate a tree')
  parser.add_argument('--delete_failure_threshold', type=int, default=25, help='max number of tries to find a node to delete')
  parser.add_argument('--subtree_selection_tries', type=int, default=50, help='max number of tries to find a subtree when inserting a binary op')
  parser.add_argument('--select_insert_loc_tries', type=int, default=10, help='max number of tries to find a insert location for a partner op')
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
