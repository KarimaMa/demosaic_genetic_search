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
import glob
import copy

import torch
import cost
from demosaic_ast import structural_hash
from mutate import Mutator
import util
from model_database import ModelDatabase 



class Searcher():
  def __init__(self, args, search_logger, debug_logger):
    self.args = args  
    self.search_logger = search_logger
    self.debug_logger = debug_logger
    self.mutator = Mutator(args, debug_logger)
    self.evaluator = cost.ModelEvaluator(args)
    self.model_manager = util.ModelManager(args.model_path)
    self.model_database = ModelDatabase(args.model_database_dir)
    
  # searches over program mutations within tiers of computational cost
  def search(self, compute_cost_tiers, tier_size):
    # initialize cost tiers
    cost_tiers = cost.CostTiers(compute_cost_tiers)
    seed_model, seed_ast = util.load_model_from_file(self.args.seed_model_file)
    # cost the initial tree
    compute_cost = self.evaluator.compute_cost(seed_model)
    model_accuracy = self.args.seed_model_accuracy
    cost_tiers.add(self.model_manager.SEED_ID, compute_cost, model_accuracy)
    
    self.model_database.add(self.model_manager.SEED_ID,\
                        [self.model_manager.SEED_ID, structural_hash(seed_ast), 1, \
                        0, [model_accuracy], compute_cost, -1])

    # CHANGE TO NOT BE FIXED - SHOULD BE INFERED FROM TASK
    model_inputs = set(("Input(Bayer)",))

    for generation in range(self.args.generations):
      new_cost_tiers = copy.deepcopy(cost_tiers) 

      for tier in cost_tiers.tiers:
        for model_id, costs in tier.items():
          best_model_version = self.model_database.get_best_version_id(model_id)
          model, model_ast = self.model_manager.load_model(model_id, best_model_version)

          new_model_ast, shash = self.mutator.mutate(model_id, model_ast, model_inputs)
          new_models = [new_model_ast.ast_to_model() for i in range(args.model_initializations)]

          # TODO
          #new_model.weight_transfer(model) # try to reuse weights from parent model

          new_model_id = self.model_manager.get_next_model_id()
          new_model_dir = self.model_manager.model_dir(new_model_id)

          util.create_dir(new_model_dir)
          perf_costs = self.evaluator.performance_cost(new_models, new_model_id, new_model_dir) # trains and evaluates model
          compute_cost = self.evaluator.compute_cost(new_model_ast)

          min_perf_cost = min(perf_costs)
          best_new_model_version = perf_costs.index(min_perf_cost)

          self.model_manager.save_model(new_models, new_model_ast, new_model_dir)
          new_cost_tiers.add(new_model_id, compute_cost, min_perf_cost)

          self.model_database.add(new_model_id,\
                        [new_model_id, shash, 1, \
                        best_new_model_version, perf_costs, compute_cost, model_id])

      new_cost_tiers.keep_topk(tier_size)
      cost_tiers = new_cost_tiers

      if generation % self.args.database_save_freq == 0:
        self.model_database.save()
        self.update_model_occurences()

    return cost_tiers

  """
  occasionally consult mutator to update model occurences 
  """
  def update_model_occurences(self):
    for model in self.mutator.seen_models:
      model_ids = self.mutator.seen_models[model]
      occurences = len(tree_ids)
      for model_id in model_ids:
        self.model_database.update_occurence_count(model_id, occurences)


def tier_range(s):
  try:
    x, y = map(int, s.split(','))
    return x, y
  except:
    raise argparse.ArgumentTypeError("Ranges must be x,y")



if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--momentum', type=float, default=90., help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
  parser.add_argument('--save_freq', type=float, default=5000, help='save frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--default_channels', type=int, default=16, help='num of output channels for conv layers')
  parser.add_argument('--max_nodes', type=int, default=30, help='max number of nodes in a tree')
  parser.add_argument('--min_subtree_size', type=int, default=2, help='minimum size of subtree in insertion')
  parser.add_argument('--max_subtree_size', type=int, default=11, help='maximum size of subtree in insertion')
  parser.add_argument('--structural_sim_reject', type=float, default=0.66, help='rejection probability threshold for structurally similar trees')
  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--model_database_dir', type=str, default='model_database', help='path to save model statistics')
  parser.add_argument('--database_save_freq', type=int, default=5, help='model database save frequency')
  parser.add_argument('--save', type=str, default='SEARCH', help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--seed_model_accuracy', type=float, help='accuracy of seed model')
  parser.add_argument('--train_portion', type=float, default=0.2, help='portion of training data')
  parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')
  parser.add_argument('--generations', type=int, help='model search generations')
  parser.add_argument('--seed_model_file', type=str, help='')
  parser.add_argument('--cost_tiers', type=tier_range, help='list of tuples of cost tier ranges', nargs=5)

  args = parser.parse_args()

  args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

  if not torch.cuda.is_available():
    search_logger.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = False
  torch.manual_seed(args.seed)

  cudnn.enabled=True
  cudnn.deterministic=True
  torch.cuda.manual_seed(args.seed)
  search_logger.info('gpu device = %d' % args.gpu)
  search_logger.info("args = %s", args)

  searcher = Searcher(args, search_logger, debug_logger)
  searcher.search(args.cost_tiers, 10)
