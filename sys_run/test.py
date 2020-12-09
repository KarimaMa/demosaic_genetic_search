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
import cost
import pareto_util
from cost import ModelEvaluator, CostTiers
from demosaic_ast import structural_hash, Downsample
from mutate import Mutator, has_downsample, MutationType
import util
from database import Database 
from monitor import Monitor, TimeoutError
from train import train_model
import torch.distributed as dist
import torch.multiprocessing as mp
import ctypes
import datetime
from job_queue import ProcessQueue
import mysql_db


def build_model_database(args):
  fields = ["model_id", "id_str", "hash", "structural_hash", "generation", "occurrences", "best_init"]
  field_types = [int, str, int, int, int, int, int]
  for model_init in range(args.model_initializations):
    fields += [f"psnr_{model_init}"]
    field_types += [float]
  fields += ["compute_cost", "parent_id", "failed_births", "failed_mutations",\
             "prune_rejections", "structural_rejections", "seen_rejections"]
  field_types += [float, int, int, int, int, int, int]

  return Database("ModelDatabase", fields, field_types, args.model_database_dir)

def build_failure_database(args):
  fields = ["model_id", "hash", "mutation_type", "insert_ops", \
            "insert_child_id", "delete_id", "node_id", "new_output_channels", "new_grouping"]
  field_types = [int, int, int, str, str, int, int, int, int, int, int]
  failure_database = Database("FailureDatabase", fields, field_types, args.failure_database_dir)
  failure_database.cntr = 0
  return failure_database

def model_database_entry(model_id, id_str, model_ast, shash, generation, compute_cost, parent_id, mutation_stats):
  data = {'model_id': model_id,
         'id_str' : id_str,
         'hash': hash(model_ast),
         'structural_hash': shash,
         'generation': generation,
         'occurrences': 1,
         'compute_cost': compute_cost,
         'parent_id': parent_id,
         'failed_births': 0,
         'failed_mutations': mutation_stats.failures,
         'prune_rejections': mutation_stats.prune_rejections,
         'structural_rejections': mutation_stats.structural_rejections,
         'seen_rejections': mutation_stats.seen_rejections}
  return data

def run_train(task_id, train_args, gpu_ids, model_id, pytorch_models, model_dir, \
              train_psnrs, valid_psnrs, log_format, task_logger):
  inits = train_args.model_initializations
  gpu_id = gpu_ids[task_id]
  try:
    for m in pytorch_models:
      m._initialize_parameters()
  except RuntimeError:
    debug_logger.debug(f"Failed to initialize model {model_id}")
    print(f"Failed to initialize model {model_id}")
  else:
    util.create_dir(model_dir)
    training_logger = util.create_logger(f'model_{model_id}_train_logger', logging.INFO, log_format, \
                                        os.path.join(model_dir, f'model_{model_id}_training_log'))
    
    print('Task ', task_id, ' launched on GPU ', gpu_id, ' model id ', model_id)
    model_valid_psnrs, model_train_psnrs = train_model(train_args, gpu_id, model_id, pytorch_models, model_dir, training_logger)
    for i in range(train_args.model_initializations):
      index = train_args.model_initializations * task_id + i 
      train_psnrs[index] = model_train_psnrs[i]
      valid_psnrs[index] = model_valid_psnrs[i]

 
def init_process(task_id, fn, train_args, gpu_ids, model_id, models, model_dir,\
                train_psnrs, valid_psnrs, log_format, task_logger, backend='nccl'):
  """ Initialize the distributed environment. """
  #os.environ['MASTER_ADDR'] = '127.0.0.1'
  #os.environ['MASTER_PORT'] = '29500'

  #dist.init_process_group(backend, rank=task_id, world_size=num_tasks)

  fn(task_id, train_args, gpu_ids, model_id, models, model_dir, \
    train_psnrs, valid_psnrs, log_format, task_logger)


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


class MutationTaskInfo:
  def __init__(self, task_id, database_entry, model_dir, models, model_id):
    self.task_id = task_id
    self.database_entry = database_entry
    self.model_dir = model_dir
    self.models = models
    self.model_id = model_id


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
      format=self.log_format, datefmt='%m/%d %I:%M:%S %p')

    self.search_logger = util.create_logger('search_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'search_log'))
    self.debug_logger = util.create_logger('debug_logger', logging.DEBUG, self.log_format, \
                                  os.path.join(args.save, 'debug_log'))
    self.mysql_logger = util.create_logger('mysql_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'myql_log'))
    self.monitor_logger = util.create_logger('monitor_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'monitor_log'))
    self.task_logger = util.create_logger('task_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'task_log'))
    self.search_logger.info("args = %s", args)

    self.args = args  
    self.mutator = Mutator(args, self.debug_logger, self.mysql_logger)
    self.evaluator = ModelEvaluator(args)
    self.model_manager = util.ModelManager(args.model_path, args.starting_model_id)

    self.model_database = build_model_database(self.args)
    self.failure_database = build_failure_database(self.args)

    # build monitors
    self.load_monitor = Monitor("Load Monitor", args.load_timeout, self.monitor_logger)
    self.mutate_monitor = Monitor("Mutation Monitor", args.mutate_timeout, self.monitor_logger)
    self.lowering_monitor = Monitor("Lowering Monitor", args.lowering_timeout, self.monitor_logger)
    self.train_monitor = Monitor("Train Monitor", args.train_timeout, self.monitor_logger)
    self.save_monitor = Monitor("Save Monitor", args.save_timeout, self.monitor_logger)

    self.mutation_batch_size = args.num_gpus
    self.mutation_batches_per_generation = int(math.ceil(args.mutations_per_generation / self.mutation_batch_size))

    mp.set_start_method("spawn", force="True")

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

  def update_failure_database(self, model_id, model_ast):
    for failure in self.mutator.failed_mutation_info:      
      failure_data = {"model_id" : model_id,
                    "hash" : hash(model_ast),
                    "mutation_type" : failure["mutation_type"]}
      
      if failure["mutation_type"] == MutationType.INSERTION.name:
        failure_data["insert_ops"] = failure["insert_ops"]
        failure_data["insert_child_id"] = failure["insert_child_id"]

        failure_data["delete_id"] = -1
        failure_data["node_id"] = -1
        failure_data["new_output_channels"] = -1
        failure_data["new_grouping"] = -1
      elif failure["mutation_type"] == MutationType.DECOUPLE.name:
        if "node_id" in failure:
          failure_data["node_id"] = failure["node_id"]
        else:
          failure_data["node_id"] = -1

        failure_data["insert_ops"] = None
        failure_data["insert_child_id"] = -1
        failure_data["delete_id"] = -1
        failure_data["new_output_channels"] = -1
        failure_data["new_grouping"] = -1
      elif failure["mutation_type"] == MutationType.CHANNEL_CHANGE.name:
        if "node_id" in failure:
          failure_data["node_id"] = failure["node_id"]
        else:
          failure_data["node_id"] = -1
        if "new_output_channels" in failure:
          failure_data["new_output_channels"] = failure["new_output_channels"]
        else:
          failure_data["new_output_channels"] = -1

        failure_data["insert_ops"] = None
        failure_data["insert_child_id"] = -1
        failure_data["delete_id"] = -1
        failure_data["new_grouping"] = -1

      elif failure["mutation_type"] == MutationType.GROUP_CHANGE.name:
        if "node_id" in failure:
          failure_data["node_id"] = failure["node_id"]
        else:
          failure_data["node_id"] = -1
        if "new_grouping" in failure:
          failure_data["new_grouping"] = failure["new_grouping"]
        else:
          failure_data["new_grouping"] = -1

        failure_data["new_output_channels"] = -1
        failure_data["insert_ops"] = None
        failure_data["insert_child_id"] = -1
        failure_data["delete_id"] = -1
      else:
        failure_data["delete_id"] = failure["delete_id"]
        failure_data["insert_ops"] = None
        failure_data["insert_child_id"] = -1
        failure_data["node_id"] = -1
        failure_data["new_output_channels"] = -1
        failure_data["new_grouping"] = -1

      self.failure_database.add(self.failure_database.cntr, failure_data)
      self.failure_database.cntr += 1

  def update_model_database(self, mutation_task_info, validation_psnrs):
    model_inits = self.args.model_initializations
    new_model_id = mutation_task_info.model_id 

    best_psnr = max(validation_psnrs)
    best_initialization = validation_psnrs.index(best_psnr)
    new_model_entry = mutation_task_info.database_entry

    new_model_entry["best_init"] = best_initialization
    for model_init in range(model_inits):
      new_model_entry[f"psnr_{model_init}"] = validation_psnrs[model_init]

    self.model_database.add(new_model_id, new_model_entry)


  def load_model(self, model_id):
    self.load_monitor.set_error_msg(f"---\nLoading model id {model_id} timed out\n---")
    with self.load_monitor:
      try:
        model_ast = self.model_manager.load_model_ast(model_id)
      except RuntimeError:
        self.load_monitor.logger.info(f"---\nError loading model {model_id}\n---")
        return None
      except TimeoutError:
        return None
      else:
        if has_loop(model_ast):
          self.debug_logger.info(f"Tree has loop!!\n{model_ast.dump()}")
          return None
        return model_ast

  def mutate_model(self, parent_id, new_model_id, model_ast, generation_stats):
    self.mutate_monitor.set_error_msg(f"---\nfailed to mutate model id {parent_id}\n---")
    with self.mutate_monitor:
      try:
        if self.args.full_model:
          model_inputs = set(("Input(Bayer)", "Input(Green)"))
        else: 
          model_inputs = set(("Input(Bayer)",))
        new_model_ast, shash, mutation_stats = self.mutator.mutate(parent_id, new_model_id, model_ast, model_inputs)
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
              
  def lower_model(self, new_model_id, new_model_ast):
    self.lowering_monitor.set_error_msg(f"---\nfailed to lower model {new_model_id}\n---")
    with self.lowering_monitor:
      try:
        new_models = [new_model_ast.ast_to_model() for i in range(self.args.model_initializations)]
      except TimeoutError:
        return None
      else:
        return new_models


  def create_train_process(self, mutation_task_info, gpu_ids, train_psnrs, validation_psnrs):
    model_id = mutation_task_info.model_id
    task_id = mutation_task_info.task_id
    #gpu_id = mutation_task_info.gpu_mapping
    model_dir = mutation_task_info.model_dir 
    models = mutation_task_info.models

    p = mp.Process(target=init_process, args=(task_id, run_train, self.args, gpu_ids, model_id, models, model_dir, \
                                              train_psnrs, validation_psnrs, self.log_format, self.task_logger))
    return p

  def run_training_tasks(self, gpu_ids, process_queue, train_psnrs, validation_psnrs):
    timeout = self.args.train_timeout
    bootup_time = 30
    available_gpus = set((0,1,2,3))
    running_processes = {}
    start_times = {}
    restarted = set()
    failed = set()

    while True:
      if process_queue.is_empty() and len(running_processes) == 0:
        break
      # check for finished tasks and kill any that have exceeded timemout
      running_tasks = [tid for tid in running_processes.keys()]
      self.task_logger.info(f"running tasks {running_tasks}")
      for task_id in running_tasks:
        task, task_info = running_processes[task_id]

        # check if process is done
        if not task.is_alive():
          self.task_logger.info(f"task {task_id} model_id {task_info.model_id} process name {task.name} is finished on time")
          self.task_logger.info(f"tasks still in queue {[tid for tid, t in process_queue.queue]}")
          task.join()
          # mark the GPU it used as free
          available_gpus.add(gpu_ids[task_id])
          del running_processes[task_id]

        else: # task is still alive, check if it timed out
          curr_time = time.time()
          start_time = start_times[task_id]
          if curr_time - start_time > timeout:
            self.task_logger.info(f"task {task_id} model_id {task_info.model_id} process name {task.name} timed out, killing at {datetime.datetime.now()}")
            task.terminate()
            task.join()
            # mark the GPU it used as free
            available_gpus.add(gpu_ids[task_id])
            del running_processes[task_id]
      
          # check if task ran into issues starting up and needs to be restarted
          if curr_time - start_time > bootup_time:
            if not os.path.exists(f"{task_info.model_dir}/v0_train_log"):
              task.terminate()
              task.join()
              if not task_id in restarted:
                self.task_logger.info(f"task {task_id} model_id {task_info.model_id} process name {task.name} " +
                                      f"unresponsive, restarting on gpu {gpu_ids[task_id]}...")
                new_task = self.create_train_process(task_info, gpu_ids, train_psnrs, validation_psnrs)
                new_task.start()
                start_times[task_id] = time.time()
                restarted.add(task_id)
              else: # we've already failed to restart this task, give up 
                failed.add(task_id)
                available_gpus.add(gpu_ids[task_id])
                del running_processes[task_id]
      
      self.task_logger.info(f"available_gpus {available_gpus}")

      # fill up any available gpus
      available_gpu_list = [g for g in available_gpus]
      for available_gpu in available_gpu_list:
        if process_queue.is_empty():
          break
        task, task_info = process_queue.take()
        task_id = task_info.task_id
        gpu_ids[task_id] = available_gpu
        self.task_logger.info(f"starting task {task_id} model_id {task_info.model_id} on gpu {available_gpu}")
        task.start()
        running_processes[task_id] = (task, task_info)
        start_times[task_id] = time.time()
        available_gpus.remove(available_gpu)

      time.sleep(20)

    return failed 


  def launch_train_processes(self, mutation_batch_info):
    # we may have fewer new models than available GPUs due to failed mutations 
    size = len(mutation_batch_info.model_ids)

    processes = []
    valid_psnrs = mp.Array(ctypes.c_double, [-1]*(size*self.args.model_initializations))
    train_psnrs = mp.Array(ctypes.c_double, [-1]*(size*self.args.model_initializations))

    rankd2modelId = {}
    for rank in range(size):
      model_id = mutation_batch_info.model_ids[rank]
      gpu_id = mutation_batch_info.gpu_mapping[model_id]
      rankd2modelId[rank] = model_id

      model_dir = mutation_batch_info.model_dirs[model_id]
      models = mutation_batch_info.pytorch_models[model_id]

      p = mp.Process(target=init_process, args=(rank, size, run_train, self.args, gpu_id, model_id, models, model_dir, \
                                                train_psnrs, valid_psnrs, self.log_format, self.debug_logger))
      p.start()
      processes.append(p)

    timeout = self.args.train_timeout
    start = time.time()
    print(f"launching training batch at {datetime.datetime.now()}")
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
      print("timed out, killing all processes still alive")
      for p in processes:
        if not p.is_alive():
          print(f'process {p.name} is finished')
        else:
          print(f'process {p.name} killed due to timeout at {datetime.datetime.now()}')
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


  def save_model_ast(self, model_ast, model_id, model_dir):
    self.save_monitor.set_error_msg(f"---\nfailed to save model {model_id}\n---")
    with self.save_monitor:
      try:
        self.model_manager.save_model_ast(model_ast, model_dir)
        self.model_manager.save_model_info_file(model_dir, self.args.model_initializations)
      except TimeoutError:
        return False
      else:
        return True

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
    cost_tiers = CostTiers(self.args.tier_database_dir, compute_cost_tiers, self.search_logger)
    if self.args.restart_generation is not None:
      cost_tiers.load_generation_from_database(self.args.tier_snapshot, self.args.restart_generation-1)

    if self.args.model_db_snapshot is not None:
      self.model_database.load(self.args.model_db_snapshot)

    seed_model, seed_ast = util.load_model_from_file(self.args.seed_model_file, self.args.seed_model_version, 0)
    seed_model_dir = self.model_manager.model_dir(self.model_manager.SEED_ID)
    util.create_dir(seed_model_dir)
    seed_ast.compute_input_output_channels()
    self.model_manager.save_model([seed_model], seed_ast, seed_model_dir)

    compute_cost = self.evaluator.compute_cost(seed_ast)
    model_accuracy = self.args.seed_model_psnr

    self.search_logger.info(f"using seed model:\n{seed_ast.dump()}\nwith cost {compute_cost} and PSNR {model_accuracy}")

    cost_tiers.add(self.model_manager.SEED_ID, compute_cost, model_accuracy)
    
    self.model_database.add(self.model_manager.SEED_ID,\
            {'model_id': self.model_manager.SEED_ID,
             'id_str' : seed_ast.id_string(),
             'hash': hash(seed_ast),
             'structural_hash': structural_hash(seed_ast),
             'occurrences': 1,
             'generation': -1,
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
    if self.args.restart_generation is None:
      start_generation = 0
      end_generation = self.args.generations 
    else:
      start_generation = self.args.restart_generation
      end_generation = start_generation + self.args.generations 

    self.search_logger.info(f"   --- STARTING SEARCH AT GENERATION {start_generation} AND ENDING AT {end_generation} ---")
    for generation in range(start_generation, end_generation):
      generational_tier_sizes = [len(tier.items()) for tier in cost_tiers.tiers]
      self.search_logger.info(f"--- STARTING GENERATION {generation} ---")
      printstr = "tier sizes: "
      for t in generational_tier_sizes:
        printstr += f"{t} "
      self.search_logger.info(printstr)
      self.search_logger.info("---------------")

      generation_stats = GenerationStats()
    
      new_cost_tiers = copy.deepcopy(cost_tiers) 

      for tid, tier in enumerate(cost_tiers.tiers):
        if len(tier) == 0:
          continue

        if self.args.pareto_sampling:
          tier_sampler = pareto_util.Sampler(tier, self.args.pareto_factor)
        else:
          tier_sampler = cost.Sampler(tier)

        self.search_logger.info(f"\n--- sampling tier {tid} size: {len(tier)} min psnr: {tier_sampler.min} max psnr: {tier_sampler.max} ---")
        
        process_queue = ProcessQueue()

        # importance sample which models from each tier to mutate based on PSNR
        model_ids = [tier_sampler.sample() for mutation in range(self.args.mutations_per_generation)]
        mutation_batch_info = MutationBatchInfo() # we'll store results from this mutation batch here          
        training_tasks = []

        size = len(model_ids) * self.args.model_initializations
        valid_psnrs = mp.Array(ctypes.c_double, [-1]*size)
        train_psnrs = mp.Array(ctypes.c_double, [-1]*size)
        gpu_ids = mp.Array(ctypes.c_int, [-1]*len(model_ids))

        for task_id, model_id in enumerate(model_ids):
          self.debug_logger.debug(f"--- loading model {model_id} ---")
          model_ast = self.load_model(model_id)

          if model_ast is None:
            continue

          new_model_id = self.model_manager.get_next_model_id()
          new_model_ast, shash, mutation_stats = self.mutate_model(model_id, new_model_id, model_ast, generation_stats)
          if new_model_ast is None:
            continue
                
          pytorch_models = self.lower_model(new_model_id, new_model_ast)
          if pytorch_models is None:
            print(f"FAILED TO LOWER MODEL {new_model_id}")
            continue

          compute_cost = self.evaluator.compute_cost(new_model_ast)
          if compute_cost > new_cost_tiers.max_cost:
            self.debug_logger.info(f"dropping model with cost {compute_cost} - too computationally expensive")
            continue

          new_model_dir = self.model_manager.model_dir(new_model_id)
          new_model_entry = model_database_entry(new_model_id, new_model_ast.id_string(), new_model_ast, \
                                                shash, generation, compute_cost, model_id, mutation_stats)

          task_info = MutationTaskInfo(task_id, new_model_entry, new_model_dir, pytorch_models, new_model_id)

          # consult mysql db for seen models on other machines
          seen_psnrs = mysql_db.find(self.args.mysql_auth, self.args.tablename, hash(new_model_ast), \
                                      new_model_ast.id_string(), self.mysql_logger)
          if not seen_psnrs is None: # model seen on other machine, skip training and use the given psnrs
            self.search_logger.info(f"model {new_model_id} already seen on another machine")
            self.update_model_database(task_info, seen_psnrs)

            util.create_dir(task_info.model_dir)
            self.save_model_ast(new_model_ast, task_info.model_id, task_info.model_dir)

            best_psnr = max(seen_psnrs)
            compute_cost = task_info.database_entry["compute_cost"]
            new_cost_tiers.add(task_info.model_id, compute_cost, best_psnr)
          else:
            training_tasks.append((new_model_ast, task_info))
        
      
        # update model database 
        for new_model_ast, task_info in training_tasks:
          model_inits = self.args.model_initializations
          task_id = task_info.task_id 
          index = task_id * model_inits

          model_psnrs = list(np.random.rand(len(model_ids)) * 8 + 25)
          self.update_model_database(task_info, model_psnrs)
          util.create_dir(task_info.model_dir)
          # training subprocess handles saving the model weights - save the ast here in the master process
          success = self.save_model_ast(new_model_ast, task_info.model_id, task_info.model_dir)
          if not success:
            continue

          # if model weights and ast were successfully saved, add model to cost tiers
          best_psnr = max(model_psnrs)
          if math.isnan(best_psnr):
            continue # don't add model to tier 

          compute_cost = task_info.database_entry["compute_cost"]
          new_cost_tiers.add(task_info.model_id, compute_cost, best_psnr)
          mysql_db.mysql_insert(self.args.mysql_auth, self.args.tablename, task_info.model_id, self.args.machine, \
                                self.args.save, hash(new_model_ast), new_model_ast.id_string(), model_psnrs, self.mysql_logger)

        self.model_database.save()

      if self.args.pareto_sampling:
        new_cost_tiers.pareto_keep_topk(tier_size)
      else:
        new_cost_tiers.keep_topk(tier_size)
      cost_tiers = new_cost_tiers

      self.update_model_occurences()
      self.model_database.save()
      self.failure_database.save()
      cost_tiers.update_database(generation)
      cost_tiers.tier_database.save()

      print(f"generation {generation} seen models {len(self.mutator.seen_models)}")
      print(f"number of killed mutations {generation_stats.generational_killed} caused by:")
      print(f"seen_rejections {generation_stats.seen_rejections} prune_rejections {generation_stats.pruned_mutations}" 
            + f" structural_rejections {generation_stats.structural_rejections} failed_mutations {generation_stats.failed_mutations}")

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

  parser.add_argument('--max_channels', type=int, default=32, help='max number of channels in any op')
  parser.add_argument('--default_channels', type=int, default=12, help='initial channel count for Convs')
  parser.add_argument('--max_nodes', type=int, default=35, help='max number of nodes in a tree')
  parser.add_argument('--min_subtree_size', type=int, default=1, help='minimum size of subtree in insertion')
  parser.add_argument('--max_subtree_size', type=int, default=12, help='maximum size of subtree in insertion')
  parser.add_argument('--structural_sim_reject', type=float, default=0.66, help='rejection probability threshold for structurally similar trees')

  parser.add_argument('--starting_model_id', type=int)
  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--model_database_dir', type=str, default='model_database', help='path to save model statistics')
  parser.add_argument('--failure_database_dir', type=str, default='failure_database', help='path to save mutation failure statistics')
  parser.add_argument('--tier_database_dir', type=str, default='cost_tier_database', help='path to save cost tier snapshot')
  parser.add_argument('--restart_generation', type=int, help='generation to start search from if restarting a prior run')
  parser.add_argument('--tier_snapshot', type=str, help='saved cost tiers to restart from')
  parser.add_argument('--model_db_snapshot', type=str, help='saved model database to restart from')
  parser.add_argument('--save', type=str, help='experiment name')

  parser.add_argument('--seed', type=int, default=1, help='random seed')

  # seed models 
  parser.add_argument('--green_seed_model_file', type=str, default='DATADUMP/GREEN_MULTIRESQUAD_SEED/models/seed/model_info', help='')
  parser.add_argument('--green_seed_model_version', type=int, default=0)
  parser.add_argument('--green_seed_model_psnr', type=float, default=32.53)

  parser.add_argument('--chroma_seed_model_file', type=str, default='DATADUMP/SIMPLE_GREEN_INPUT_CHROMA_MODEL/models/seed/model_info', help='')
  parser.add_argument('--chroma_seed_model_version', type=int, default=0)
  parser.add_argument('--chroma_seed_model_psnr', type=float, default=32.32)

  parser.add_argument('--generations', type=int, default=20, help='model search generations')
  parser.add_argument('--cost_tiers', type=str, help='list of tuples of cost tier ranges')
  parser.add_argument('--tier_size', type=int, default=15, help='how many models to keep per tier')
  parser.add_argument('--pareto_sampling', action='store_true', help='whether to use pareto sampling')
  parser.add_argument('--pareto_factor', type=float, help='discount factor per pareto frontier')
  parser.add_argument('--mutations_per_generation', type=int, default=12, help='how many mutations produced by each tier per generation')

  parser.add_argument('--mutation_failure_threshold', type=int, default=500, help='max number of tries to mutate a tree')
  parser.add_argument('--delete_failure_threshold', type=int, default=25, help='max number of tries to find a node to delete')
  parser.add_argument('--subtree_selection_tries', type=int, default=50, help='max number of tries to find a subtree when inserting a binary op')
  parser.add_argument('--select_insert_loc_tries', type=int, default=10, help='max number of tries to find a insert location for a partner op')
  
  parser.add_argument('--load_timeout', type=int, default=10)
  parser.add_argument('--mutate_timeout', type=int, default=30)
  parser.add_argument('--lowering_timeout', type=int, default=10)
  parser.add_argument('--train_timeout', type=int, default=2400)
  parser.add_argument('--save_timeout', type=int, default=10)

  # training parameters
  parser.add_argument('--num_gpus', type=int, default=4, help='number of available GPUs') # change this to use all available GPUs
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
  parser.add_argument('--lr_search_steps', type=int, default=1, help='how many line search iters for finding learning rate')
  parser.add_argument('--variance_min', type=float, default=0.003, help='minimum validation psnr variance')
  parser.add_argument('--variance_max', type=float, default=0.02, help='maximum validation psnr variance')
  parser.add_argument('--weight_decay', type=float, default=1e-16, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=200, help='training report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='trained weights save frequency')
  parser.add_argument('--epochs', type=int, default=3, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
  parser.add_argument('--train_portion', type=int, default=1e5, help='portion of training data to use')
  parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')
  parser.add_argument('--validation_freq', type=int, default=50, help='validation frequency for assessing validation PSNR variance')

  # training full chroma + green parameters
  parser.add_argument('--use_green_input', action="store_true")
  parser.add_argument('--full_model', action="store_true")

  parser.add_argument('--mysql_auth', type=str)
  parser.add_argument("--machine", type=str)

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

  if args.full_model:
    args.seed_model_file = args.chroma_seed_model_file
    args.seed_model_version = args.chroma_seed_model_version
    args.seed_model_psnr = args.chroma_seed_model_psnr
    args.tablename = "chroma"
    args.task_out_c = 6
  else:
    args.seed_model_file = args.green_seed_model_file
    args.seed_model_version = args.green_seed_model_version
    args.seed_model_psnr = args.green_seed_model_psnr
    args.tablename = "green"
    args.task_out_c = 2

  args.cost_tiers = parse_cost_tiers(args.cost_tiers)
  util.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  args.model_path = os.path.join(args.save, args.model_path)
  util.create_dir(args.model_path)
  args.model_database_dir = os.path.join(args.save, args.model_database_dir)
  util.create_dir(args.model_database_dir)
  args.failure_database_dir = os.path.join(args.save, args.failure_database_dir)
  util.create_dir(args.failure_database_dir)
  args.tier_database_dir = os.path.join(args.save, args.tier_database_dir)
  util.create_dir(args.tier_database_dir)

  searcher = Searcher(args)
  searcher.search(args.cost_tiers, args.tier_size)
