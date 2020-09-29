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
import math 
import logging
import sys
import asyncio
import time
import os
import argparse
import glob
import copy
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import sys
sys.path.append(sys.path[0].split("/")[0])
from tree import has_loop
from cost import ModelEvaluator, CostTiers, Sampler
from demosaic_ast import structural_hash, Downsample
from mutate import Mutator, has_downsample, MutationType
import util
from database import Database 
import meta_model
from model_lib import multires_green_model, multires_green_model2, multires_green_model3
from util import PerfStatTracker
from type_check import check_channel_count, shrink_channels
from monitor import Monitor, TimeoutError
from train import train_model
from dataset import GreenDataset, ids_from_file
import torch.distributed as dist
import torch.multiprocessing as mp
import ctypes

def create_validation_dataset(args):
  validation_data = GreenDataset(data_file=args.validation_file, RAM=True)
  num_validation = len(validation_data)
  validation_indices = list(range(num_validation))

  validation_queue = torch.utils.data.DataLoader(
      validation_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
      pin_memory=True, num_workers=0)
  return validation_queue


def create_train_dataset(args):
  full_data_filenames = ids_from_file(args.training_file)
  used_filenames = full_data_filenames[0:int(args.train_portion)]
  train_data = GreenDataset(data_filenames=used_filenames, RAM=True) 

  num_train = len(train_data)
  train_indices = list(range(num_train))
  
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=True, num_workers=0)
  return train_queue


def build_model_database(args):
  fields = ["model_id", "id_str", "hash", "structural_hash", "occurrences", "best_init"]
  field_types = [int, int, int, int, int]
  for model_init in range(args.model_initializations):
    fields += [f"psnr_{model_init}"]
    field_types += [float]
  fields += ["compute_cost", "parent_id", "failed_births", "failed_mutations",\
             "prune_rejections", "structural_rejections", "seen_rejections"]
  field_types += [float, int, int, int, int, int, int]

  return Database("ModelDatabase", fields, field_types, args.model_database_dir)

def build_failure_database(args):
  fields = ["model_id", "hash", "mutation_type", "insert_ops", \
            "insert_child_id", "delete_id"]
  field_types = [int, int, int, str, str, int, int, int]
  failure_database = Database("FailureDatabase", fields, field_types, args.failure_database_dir)
  failure_database.cntr = 0
  return failure_database

def model_database_entry(model_id, id_str, model_ast, shash, compute_cost, parent_id, mutation_stats):
  data = {'model_id': model_id,
         'id_str' : id_str,
         'hash': hash(model_ast),
         'structural_hash': shash,
         'occurrences': 1,
         'compute_cost': compute_cost,
         'parent_id': parent_id,
         'failed_births': 0,
         'failed_mutations': mutation_stats.failures,
         'prune_rejections': mutation_stats.prune_rejections,
         'structural_rejections': mutation_stats.structural_rejections,
         'seen_rejections': mutation_stats.seen_rejections}
  return data

def run_train(rank, train_args, gpu_id, train_data, valid_data, model_id, \
            models, model_dir, train_psnrs, valid_psnrs, log_format, debug_logger):

  try:
    for m in models:
      m._initialize_parameters()
  except RuntimeError:
    debug_logger.debug(f"Failed to initialize model {model_id}")
  else:
    util.create_dir(model_dir)
    training_logger = util.create_logger(f'model_{model_id}_train_logger', logging.INFO, log_format, \
                                        os.path.join(model_dir, f'model_{model_id}_training_log'))
    
    print('Process ', dist.get_rank(), ' launched on GPU ', gpu_id, ' model id ', model_id)
    model_valid_psnrs, model_train_psnrs = train_model(train_args, gpu_id, train_data, valid_data, \
                model_id, models, model_dir, training_logger)
    for i in range(train_args.model_initializations):
      index = train_args.model_initializations * rank + i 
      train_psnrs[index] = model_train_psnrs[i]
      valid_psnrs[index] = model_valid_psnrs[i]

 
def init_process(rank, size, fn, train_args, gpu_id, train_data, valid_data, model_id, models, model_dir,\
                train_psnrs, valid_psnrs, log_format, debug_logger, backend='gloo'):
  """ Initialize the distributed environment. """
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'

  dist.init_process_group(backend, rank=rank, world_size=size)
  fn(rank, train_args, gpu_id, train_data, valid_data, model_id, models, model_dir, \
    train_psnrs, valid_psnrs, log_format, debug_logger)


class GenerationStats():
  def __init__(self):
    self.generational_killed = 0
    self.pruned_mutations = 0
    self.failed_mutations = 0
    self.seen_rejections = 0
    self.structural_rejections = 0

  def update(self, mutation_stats):
    self.pruned_mutations += mutation_stats.prune_rejections
    self.failed_mutations += mutation_stats.failures
    self.structural_rejections += mutation_stats.structural_rejections
    self.seen_rejections += mutation_stats.seen_rejections 

  def increment_killed(self):
    self.generational_killed += 1


class MutationBatchInfo:
  def __init__(self):
    self.validation_psnrs = {}
    self.train_psnrs = {}
    self.database_entries = {}
    self.model_dirs = {}
    self.model_asts = {}
    self.pytorch_models = {}
    self.gpu_mapping = {}
    self.model_ids = []

  def add_model(self, model_id, gpu_id, models, model_dir, model_ast, model_data):
    self.pytorch_models[model_id] = models
    self.model_dirs[model_id] = model_dir
    self.model_asts[model_id] = model_ast
    self.model_ids.append(model_id)
    self.database_entries[model_id] = model_data
    self.gpu_mapping[model_id] = gpu_id

  def update_model_perf(self, model_id, validation_psnr, train_psnr):
    self.validation_psnrs[model_id] = validation_psnr
    self.train_psnrs[model_id] = train_psnr


class Searcher():
  def __init__(self, args):
    # build loggers
    self.log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
      format=log_format, datefmt='%m/%d %I:%M:%S %p')

    self.search_logger = util.create_logger('search_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'search_log'))
    self.debug_logger = util.create_logger('debug_logger', logging.DEBUG, self.log_format, \
                                  os.path.join(args.save, 'debug_log'))
    self.monitor_logger = util.create_logger('monitor_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'monitor_logger'))
    self.search_logger.info("args = %s", args)

    self.args = args  
    self.mutator = Mutator(args, debug_logger)
    self.evaluator = ModelEvaluator(args)
    self.model_manager = util.ModelManager(args.model_path)

    self.model_database = build_model_database(self.args)
    self.failure_database = build_failure_database(self.args)

    # build monitors
    self.load_monitor = Monitor("Load Monitor", args.load_timeout, self.monitor_logger)
    self.mutate_monitor = Monitor("Mutation Monitor", args.mutate_timeout, self.monitor_logger)
    self.lowering_monitor = Monitor("Lowering Monitor", args.lowering_timeout, self.monitor_logger)
    self.train_monitor = Monitor("Train Monitor", args.train_timeout, self.monitor_logger)
    self.save_monitor = Monitor("Save Monitor", args.save_timeout, self.monitor_logger)

    self.mutation_batch_size = self.args.num_gpus
    self.mutation_batches_per_generation = int(math.ceil(args.mutations_per_generation / self.mutation_batch_size))
    self.train_datasets = [create_train_dataset(args) for g in range(args.num_gpus)]
    self.valid_datasets = [create_validation_dataset(args) for g in range(args.num_gpus)]

    mp.set_start_method("spawn")


  def update_failure_database(self, model_id, model_ast):
    for failure in self.mutator.failed_mutation_info:
      failure_data = {"model_id" : model_id,
                    "hash" : hash(model_ast),
                    "mutation_type" : failure["mutation_type"]}
      if failure["mutation_type"] == MutationType.INSERTION.name:
        failure_data["insert_ops"] = failure["insert_ops"]
        failure_data["insert_child_id"] = failure["insert_child_id"]
        failure_data["delete_id"] = -1
      else:
        failure_data["insert_ops"] = None
        failure_data["insert_child_id"] = -1 
        failure_data["delete_id"] = failure["delete_id"]
    
      self.failure_database.add(self.failure_database.cntr, failure_data)
      self.failure_database.cntr += 1

  def update_model_database(self, mutation_batch_info, new_model_id):
    # update model database
    perf_costs = mutation_batch_info.validation_psnrs[new_model_id]
    min_perf_cost = min(perf_costs)
    best_new_model_initialization = perf_costs.index(min_perf_cost)
    new_model_entry = mutation_batch_info.database_entries[new_model_id]

    new_model_entry["best_init"] = best_new_model_initialization
    for model_init in range(self.args.model_initializations):
      new_model_entry[f"psnr_{model_init}"] = perf_costs[model_init]

    self.model_database.add(new_model_id, new_model_entry)

  def load_model(self, model_id):
    self.load_monitor.set_error_msg(f"---\nLoading model id {model_id} timed out\n---")
    with self.load_monitor:
      try:
        best_model_initialization = self.model_database.get(model_id, "best_init")
        model, model_ast = self.model_manager.load_model(model_id, best_model_initialization)
      except RuntimeError:
        self.load_monitor.logger.info(f"---\nError loading model {model_id}\n---")
        return None, None
      except TimeoutError:
        return None, None
      else:
        if has_loop(model_ast):
          self.debug_logger.info(f"Tree has loop!!\n{model_ast.dump()}")
          return None, None
        return model, model_ast

  def mutate_model(self, parent_id, new_model_id, model_ast, generation_stats):
    self.mutate_monitor.set_error_msg(f"---\nfailed to mutate model id {parent_id}\n---")
    with self.mutate_monitor:
      try:
        model_inputs = set(("Input(Bayer)",))
        new_model_ast, shash, mutation_stats = self.mutator.mutate(new_model_id, model_ast, model_inputs)
        generation_stats.update(mutation_stats)

        if new_model_ast is None: 
          self.model_database.increment(parent_id, 'failed_births')
          self.update_failure_database(parent_id, model_ast)
          generation_stats.increment_killed()
          return None, None, None
      except TimeoutError:
        return None, None, None
      else:
        return new_model_ast, shash, mutation_stats
              
  def lower_model(self, new_model_id, new_model_ast, gpu_id):
    self.lowering_monitor.set_error_msg(f"---\nfailed to lower model {new_model_id}\n---")
    with self.lowering_monitor:
      try:
        print(f"GPU ID {gpu_id}")
        new_models = [new_model_ast.ast_to_model(gpu_id) for i in range(self.args.model_initializations)]
      except TimeoutError:
        return None
      else:
        return new_models

  async def await_training(self, mutation_batch_info):
    print("inside await_training training")
    training_tasks = {}   
    for new_model_id in mutation_batch_info.model_ids:
      try:
        new_models = mutation_batch_info.pytorch_models[new_model_id]
        gpu_id = mutation_batch_info.gpu_mapping[new_model_id]
        new_model_dir = mutation_batch_info.model_dirs[new_model_id]

        for m in new_models:
          m._initialize_parameters()
      except RuntimeError:
        self.debug_logger.debug(f"Failed to initialize model {new_model_id}")
        continue
      else:
        util.create_dir(new_model_dir)
        training_logger = util.create_logger(f'model_{new_model_id}_train_logger', logging.INFO, self.log_format, \
                                            os.path.join(new_model_dir, f'model_{new_model_id}_training_log'))
        training_tasks[new_model_id] = asyncio.create_task(
                                          train_model(self.args, gpu_id, self.train_datasets[gpu_id], \
                                                      self.valid_datasets[gpu_id], new_model_id, \
                                                      new_models, new_model_dir, training_logger))

    print(training_tasks.values())
    done, notdone = await asyncio.wait(training_tasks.values(), timeout=self.args.train_timeout)
    return done, notdone


  def launch_train_processes(self, mutation_batch_info):
    # we may have fewer new models than available GPUs due to failed mutations 
    size = len(mutation_batch_info.model_ids)
    print(f"number of new model_ids {len(mutation_batch_info.model_ids)}")

    processes = []
    valid_psnrs = mp.Array(ctypes.c_double, [-1]*(size*self.args.model_initializations))
    train_psnrs = mp.Array(ctypes.c_double, [-1]*(size*self.args.model_initializations))
    print(valid_psnrs)

    rankd2modelId = {}
    for rank in range(size):
      model_id = mutation_batch_info.model_ids[rank]
      gpu_id = mutation_batch_info.gpu_mapping[model_id]
      rankd2modelId[rank] = model_id
      train_data = self.train_datasets[rank]
      valid_data = self.valid_datasets[rank]

      model_dir = mutation_batch_info.model_dirs[model_id]
      models = mutation_batch_info.pytorch_models[model_id]

      p = mp.Process(target=init_process, args=(rank, size, run_train, self.args, gpu_id, train_data, \
            valid_data, model_id, models, model_dir, train_psnrs, valid_psnrs, log_format, self.debug_logger))

      p.start()
      processes.append(p)

    timeout = self.args.train_timeout
    start = time.time()
    while time.time() - start <= timeout:
      if any(p.is_alive() for p in processes):
        time.sleep(10)  # Just to avoid hogging the CPU
      else:
        print('All training processes finished on time!')
        for p in processes:
          p.join() # make sure things are stopped properly
          print('stopping process {}'.format(p.name))
        break
    else:
      # We only enter this if we didn't 'break' above during the while loop!
      print("timed out, killing all processes")
      for p in processes:
        if not p.is_alive():
          print(f'process {p.name} is finished')
        else:
          print(f'process {p.name} killed due to timeout')
          p.terminate()
        print(f' -> stopping (joining) process {p.name}')
        p.join()

    for rank in range(size):
      model_id = rankd2modelId[rank]
      model_validation_psnrs = []
      model_train_psnrs = []
      for i in range(self.args.model_initializations):
        index = rank * self.args.model_initializations + i
        model_validation_psnrs.append(valid_psnrs[index])
        model_train_psnrs.append(train_psnrs[index])
      mutation_batch_info.validation_psnrs[model_id] = model_validation_psnrs
      mutation_batch_info.train_psnrs[model_id] = model_train_psnrs


  def save_model(self, mutation_batch_info, new_model_id):
    self.save_monitor.set_error_msg(f"---\nfailed to save model {new_model_id}\n---")
    with self.save_monitor:
      try:
        pytorch_models = mutation_batch_info.pytorch_models[new_model_id]
        model_ast = mutation_batch_info.model_asts[new_model_id]
        model_dir = mutation_batch_info.model_dirs[new_model_id]
        self.model_manager.save_model(pytorch_models, model_ast, model_dir)
      except TimeoutError:
        return False
      else:
        return True


  # searches over program mutations within tiers of computational cost
  def search(self, compute_cost_tiers, tier_size):
    # HACK FOR NOW  
    gpu_id = 3

    cost_tiers = CostTiers(compute_cost_tiers)

    seed_model, seed_ast = util.load_model_from_file(self.args.seed_model_file, self.args.seed_model_version, gpu_id)
    seed_model_dir = self.model_manager.model_dir(self.model_manager.SEED_ID)
    util.create_dir(seed_model_dir)
    seed_ast.compute_input_output_channels()
    self.model_manager.save_model([seed_model], seed_ast, seed_model_dir)

    seed_ast.compute_input_output_channels()
    self.search_logger.info(f"using seed model:\n{seed_ast.dump()}")

    compute_cost = self.evaluator.compute_cost(seed_ast)
    model_accuracy = self.args.seed_model_psnr

    cost_tiers.add(self.model_manager.SEED_ID, compute_cost, model_accuracy)
    
    self.model_database.add(self.model_manager.SEED_ID,\
            {'model_id': self.model_manager.SEED_ID,
             'id_str' : seed_ast.id_string(),
             'hash': hash(seed_ast),
             'structural_hash': structural_hash(seed_ast),
             'occurrences': 1,
             'best_init': 0,
             'psnr_0': model_accuracy,
             'psnr_1': -1,
             'psnr_2': -1,
             'compute_cost': compute_cost,
             'parent_id': -1,
             'failed_births': 0,
             'failed_mutations': 0,
             'prune_rejections': 0,
             'structural_rejections': 0,
             'seen_rejections': 0})

    # CHANGE TO NOT BE FIXED - SHOULD BE INFERED FROM TASK

    for generation in range(self.args.generations):
      generational_tier_sizes = [len(tier.items()) for tier in cost_tiers.tiers]
      self.search_logger.info("---------------")
      printstr = "tier sizes: "
      for t in generational_tier_sizes:
        printstr += f"{t} "
      self.search_logger.info(printstr)
      self.search_logger.info("---------------")

      generation_stats = GenerationStats()
     
      new_cost_tiers = copy.deepcopy(cost_tiers) 
      avg_iter_time = 0 

      for tid, tier in enumerate(cost_tiers.tiers):
        if len(tier) == 0:
          continue

        # importance sample which models from each tier to mutate based on PSNR
        tier_sampler = Sampler(tier)
        self.search_logger.info(f"\n--- sampling tier {tid} size: {len(tier)} min psnr: {tier_sampler.min} max psnr: {tier_sampler.max} ---")

        for mutation_batch in range(self.mutation_batches_per_generation):
          model_ids = [tier_sampler.sample() for mutation in range(self.mutation_batch_size)]
          mutation_batch_info = MutationBatchInfo() # we'll store results from this mutation batch here          
          training_tasks = {}

          for gpu_id, model_id in enumerate(model_ids):
            model, model_ast = self.load_model(model_id)
            if model_ast is None:
              continue

            new_model_id = self.model_manager.get_next_model_id()
            new_model_ast, shash, mutation_stats = self.mutate_model(model_id, new_model_id, model_ast, generation_stats)
            if new_model_ast is None:
              continue
                  
            compute_cost = self.evaluator.compute_cost(new_model_ast)
            if compute_cost > new_cost_tiers.max_cost:
              self.debug_logger.info(f"dropping model with cost {compute_cost} - too computationally expensive")
              continue

            new_models = self.lower_model(new_model_id, new_model_ast, gpu_id)
            if new_models is None:
              continue

            new_model_dir = self.model_manager.model_dir(new_model_id)
            new_model_entry = model_database_entry(new_model_id, new_model_ast.id_string(), new_model_ast, shash, compute_cost, model_id, mutation_stats)
            mutation_batch_info.add_model(new_model_id, gpu_id, new_models, new_model_dir, new_model_ast, new_model_entry)
          
          # wait for all training tasks to finish
          #done, notdone = asyncio.run(self.await_training(mutation_batch_info))
          self.launch_train_processes(mutation_batch_info)

          # finished_ids = []
          # for task in done:
          #   new_model_id, validation_psnr, train_psnr = task.result()
          #   mutation_batch_info.update_model_perf(new_model_id, validation_psnr, train_psnr)
          #   finished_ids.append(new_model_id)

          # for new_model_id in training_tasks:
          #   if not new_model_id in finished_ids:
          #     self.monitor_logger.info(f"---\nfailed to train model {new_model_id}\n---")

          # update model database with models that finished training 
          for new_model_id in mutation_batch_info.model_ids: #finished_ids:
            self.update_model_database(mutation_batch_info, new_model_id)
            success = self.save_model(mutation_batch_info, new_model_id)
            if not success:
              continue
            # if model weights and ast were successfully saved, add model to cost tiers
            min_perf_cost = min(mutation_batch_info.validation_psnrs[new_model_id])
            if math.isnan(min_perf_cost):
              continue # don't add model to tier 
            new_cost_tiers.add(new_model_id, new_model_entry['compute_cost'], min_perf_cost)

          self.model_database.save()

      new_cost_tiers.keep_topk(tier_size)
      cost_tiers = new_cost_tiers

      if generation % self.args.database_save_freq == 0:
        self.update_model_occurences()
        self.model_database.save()
        self.failure_database.save()

      print(f"generation {generation} seen models {len(self.mutator.seen_models)}")
      print(f"number of killed mutations {generation_stats.generational_killed} caused by:")
      print(f"seen_rejections {generation_stats.seen_rejections} prune_rejections {generation_stats.pruned_mutations}" 
            + f" structural_rejections {generation_stats.structural_rejections} failed_mutations {generation_stats.failed_mutations}")

    self.update_model_occurences()
    self.model_database.save()
    self.failure_database.save()

    print(self.model_database.table[10])
    # try re-loading model database
    self.model_database.load(self.model_database.database_path)
    print("--- re-loaded database ---")
    print(self.model_database.table[10])
    return cost_tiers

  """
  occasionally consult mutator to update model occurences 
  """
  def update_model_occurences(self):
    for model, value in self.mutator.seen_models.items():
      model_ids = self.mutator.seen_models[model]
      occurences = len(model_ids)
      for model_id in model_ids:
        self.model_database.update(model_id, 'occurences', occurences)


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
  parser.add_argument('--failure_database_dir', type=str, default='failure_database', help='path to save mutation failure statistics')
  parser.add_argument('--database_save_freq', type=int, default=5, help='model database save frequency')
  parser.add_argument('--save', type=str, default='MODEL_SEARCH', help='experiment name')

  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--seed_model_file', type=str, default='DATADUMP/BASIC_GREEN_SEED_5NEG3_LR/models/seed/model_info', help='')
  parser.add_argument('--seed_model_version', type=int, default=2)
  parser.add_argument('--seed_model_psnr', type=float, default=31.38)

  parser.add_argument('--generations', type=int, default=20, help='model search generations')
  parser.add_argument('--cost_tiers', type=str, help='list of tuples of cost tier ranges')
  parser.add_argument('--tier_size', type=int, default=15, help='how many models to keep per tier')
  parser.add_argument('--mutations_per_generation', type=int, default=15, help='how many mutations produced by each tier per generation')

  parser.add_argument('--mutation_failure_threshold', type=int, default=500, help='max number of tries to mutate a tree')
  parser.add_argument('--delete_failure_threshold', type=int, default=25, help='max number of tries to find a node to delete')
  parser.add_argument('--subtree_selection_tries', type=int, default=50, help='max number of tries to find a subtree when inserting a binary op')
  parser.add_argument('--select_insert_loc_tries', type=int, default=10, help='max number of tries to find a insert location for a partner op')
  
  parser.add_argument('--load_timeout', type=int, default=10)
  parser.add_argument('--mutate_timeout', type=int, default=30)
  parser.add_argument('--lowering_timeout', type=int, default=10)
  parser.add_argument('--train_timeout', type=int, default=1800)
  parser.add_argument('--save_timeout', type=int, default=10)

  # training parameters
  parser.add_argument('--num_gpus', type=int, default=1, help='number of available GPUs') # change this to use all available GPUs
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
  parser.add_argument('--lr_search_steps', type=int, default=1, help='how many line search iters for finding learning rate')
  parser.add_argument('--variance_min', type=float, default=0.003, help='minimum validation psnr variance')
  parser.add_argument('--variance_max', type=float, default=0.02, help='maximum validation psnr variance')
  parser.add_argument('--weight_decay', type=float, default=1e-10, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=200, help='training report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='trained weights save frequency')
  parser.add_argument('--epochs', type=int, default=3, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--train_portion', type=int, default=1e5, help='portion of training data to use')
  parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')
  parser.add_argument('--validation_freq', type=int, default=50, help='validation frequency for assessing validation PSNR variance')

  args = parser.parse_args()

  if not torch.cuda.is_available():
    sys.exit(1)

  random.seed(args.seed)
  np.random.seed(args.seed)
  #torch.cuda.set_device(args.gpu)
  cudnn.benchmark = False
  torch.manual_seed(args.seed)

  cudnn.enabled=True
  cudnn.deterministic=True
  torch.cuda.manual_seed(args.seed)

  args.cost_tiers = parse_cost_tiers(args.cost_tiers)
  util.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  args.model_path = os.path.join(args.save, args.model_path)
  util.create_dir(args.model_path)
  args.model_database_dir = os.path.join(args.save, args.model_database_dir)
  util.create_dir(args.model_database_dir)
  args.failure_database_dir = os.path.join(args.save, args.failure_database_dir)
  util.create_dir(args.failure_database_dir)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  search_logger = util.create_logger('search_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'search_log'))
  debug_logger = util.create_logger('debug_logger', logging.DEBUG, log_format, \
                                os.path.join(args.save, 'debug_log'))

  search_logger.info("args = %s", args)

  searcher = Searcher(args)
  searcher.search(args.cost_tiers, args.tier_size)
