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
from orderedset import OrderedSet
import torch
import sys
sys.path.append(sys.path[0].split("/")[0])
from tree import has_loop
import cost
import pareto_util
from cost import ModelEvaluator, CostTiers
import demosaic_ast
from demosaic_ast import get_green_model_id, set_green_model_id
from mutate import Mutator, has_downsample, MutationType
import util
from database import Database 
from monitor import Monitor, TimeoutError
from train import train_model
import torch.distributed as dist
import torch.multiprocessing as mp
import ctypes
import datetime
import mysql_db
# from dataset import bayer
# from imageio import imread
#from multiprocessing import shared_memory
from queue import Empty
import psutil


def build_model_database(args):
  fields = ["model_id", "id_str", "hash", "structural_hash", "generation", "occurrences", "best_init"]
  field_types = [int, str, int, int, int, int, int]
  for model_init in range(args.keep_initializations):
    fields += [f"psnr_{model_init}"]
    field_types += [float]
  fields += ["compute_cost", "parent_id", "failed_births", "failed_mutations",\
             "prune_rejections", "structural_rejections", "seen_rejections"]
  field_types += [float, int, int, int, int, int, int]

  fields += ["mutation_type", "insert_ops", "binop_operand_choice", "mutation_node_id", "new_output_channels", "new_grouping", "green_model_id"]
  field_types += [str, str, str, int, int, int, int]

  return Database("ModelDatabase", fields, field_types, args.model_database_dir)

def build_failure_database(args):
  fields = ["model_id", "hash", "mutation_type", "insert_ops", "binop_operand_choice", \
            "node_id", "new_output_channels", "new_grouping", "green_model_id"]
  field_types = [int, int, str, str, str, int, int, int, int]
  failure_database = Database("FailureDatabase", fields, field_types, args.failure_database_dir)
  failure_database.cntr = 0
  return failure_database

def build_perf_database(args):
  fields = ["model_id", "cost", "train_time"]
  field_types = [int, float, int]
  perf_database = Database("PerformanceDatabase", fields, field_types, args.performance_database_dir)
  perf_database.cntr = 0
  return perf_database


def model_database_entry(model_id, id_str, model_ast, shash, generation, \
                        compute_cost, parent_id, mutation_stats):
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
         'seen_rejections': mutation_stats.seen_rejections,
         'mutation_type': mutation_stats.used_mutation.mutation_type,
         'insert_ops': mutation_stats.used_mutation.insert_ops,
         'binop_operand_choice': mutation_stats.used_mutation.binop_operand_choice,
         'mutation_node_id': mutation_stats.used_mutation.node_id,
         'new_output_channels': mutation_stats.used_mutation.new_output_channels,
         'new_grouping': mutation_stats.used_mutation.new_grouping,
         'green_model_id': mutation_stats.used_mutation.green_model_id}
  return data



"""
loads dataset into RAM
"""
# def preload_dataset(data_file):
#   image_files = util.ids_from_file(data_file) # patch filenames
#   dataset = [None for i in range(len(image_files))]
#   imsize = 128*128*3
#   num_images = len(image_files)
#   nfloats = imsize * num_images

#   shm = shared_memory.SharedMemory(create=True, size=(nfloats * 4))
#   dataset = np.ndarray((num_images, 3, 128, 128), dtype=np.float32, buffer=shm.buf)

#   for i, image_f in enumerate(image_files):
#     if i % 1000 == 0:
#       print(i)
#     img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
#     img = np.transpose(img, [2, 0, 1])
#     if i == 0:
#       print(img[:,0:3,0:3])
    
#     dataset[i] = img

#   return shm.name


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

  def update_model_perf(self, model_id, validation_psnr):
    self.validation_psnrs[model_id] = validation_psnr


"""
 worker subprocess function that runs training
 gpu_id: the gpu assigned to this worker
"""
def run_train(work_q, worker_id, pending_start_times, pending_tasks, pending_l, finished_tasks, \
              gpu_id, args, validation_psnrs, log_format, logger):
  # keep pulling from work queue
  while True:
    try:
      train_task_info = work_q.get(block=False)
      if train_task_info is None:
        return

      model_id = train_task_info["model_id"]
      task_id = train_task_info["task_id"]
      model_dir = train_task_info["model_dir"]
      models = train_task_info["models"]

      pending_l.acquire()
      pending_start_times[worker_id] = time.time()
      pending_tasks[worker_id] = task_id
      pending_l.release()

      logger.info(f'---worker on gpu {gpu_id} running task {task_id} model {model_id} ---')
      print(f'--- worker on gpu {gpu_id} running model {model_id} with {len([p for p in models[0].parameters()])} params ---')
    
      for m in models:
        m._initialize_parameters()
     
      util.create_dir(model_dir)
      train_logger = util.create_logger(f'model_{model_id}_train_logger', logging.INFO, log_format, \
                                  os.path.join(model_dir, f'model_{model_id}_training_log'))

      model_val_psnrs = train_model(args, gpu_id, model_id, models, model_dir, train_logger)
      
      logger.info(f"worker on gpu {gpu_id} with task {task_id} writing to validation psnrs")

      for i in range(args.keep_initializations):
        index = args.keep_initializations * task_id + i 
        validation_psnrs[index] = model_val_psnrs[i]
        print(f"worker on gpu {gpu_id} task {task_id} psnr {model_val_psnrs[i]}")

      pending_l.acquire()
      start_time = pending_start_times[worker_id]
      pending_start_times[worker_id] = -1
      pending_tasks[worker_id] = -1
      pending_l.release()
      #work_q.task_done()

      print(f"task {task_id} took {time.time() - start_time}")
      finished_tasks[task_id] = time.time() - start_time # store time task took

    except Empty:
      time.sleep(10)



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
                                  os.path.join(args.save, 'mysql_log'))
    self.monitor_logger = util.create_logger('monitor_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'monitor_log'))
    self.work_manager_logger = util.create_logger('work_manager_logger', logging.INFO, self.log_format, \
                                  os.path.join(args.save, 'work_manager_log'))
    self.search_logger.info("args = %s", args)

    self.args = args  
    self.mutator = Mutator(args, self.debug_logger, self.mysql_logger)
    self.evaluator = ModelEvaluator(args)
    self.model_manager = util.ModelManager(args.model_path, args.starting_model_id)

    self.model_database = build_model_database(self.args)
    self.failure_database = build_failure_database(self.args)
    self.perf_database = build_perf_database(self.args)

    # build monitors
    self.load_monitor = Monitor("Load Monitor", args.load_timeout, self.monitor_logger)
    self.mutate_monitor = Monitor("Mutation Monitor", args.mutate_timeout, self.monitor_logger)
    self.lowering_monitor = Monitor("Lowering Monitor", args.lowering_timeout, self.monitor_logger)
    self.train_monitor = Monitor("Train Monitor", args.train_timeout, self.monitor_logger)
    self.save_monitor = Monitor("Save Monitor", args.save_timeout, self.monitor_logger)
    
    mp.set_start_method("spawn", force=True)

    self.num_workers = self.args.num_gpus
    self.models_per_gen = self.args.mutations_per_generation * self.args.keep_initializations * len(args.cost_tiers)
    self.work_queue = mp.Queue(self.models_per_gen + self.num_workers)
    self.pending_locks = [mp.Lock() for w in range(self.num_workers)]
    self.work_loggers = [util.create_logger('work_logger', logging.INFO, self.log_format, \
                        os.path.join(args.save, f'work_log_{i}')) for i in range(self.num_workers)]


    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

  def update_failure_database(self, model_id, model_ast):
    for failure in self.mutator.failed_mutation_info:      
      failure_data = {"model_id" : model_id,
                    "hash" : hash(model_ast),
                    "mutation_type" : failure.mutation_type,
                    "insert_ops" : failure.insert_ops,
                    "binop_operand_choice": failure.binop_operand_choice,
                    "node_id" : failure.node_id,
                    "new_output_channels" : failure.new_output_channels,
                    "new_grouping" : failure.new_grouping,
                    "green_model_id": failure.green_model_id}

      self.failure_database.add(self.failure_database.cntr, failure_data)
      self.failure_database.cntr += 1

  def update_model_database(self, mutation_task_info, validation_psnrs):
    model_inits = self.args.keep_initializations
    new_model_id = mutation_task_info.model_id 

    best_psnr = max(validation_psnrs)
    best_initialization = validation_psnrs.index(best_psnr)
    new_model_entry = mutation_task_info.database_entry

    new_model_entry["best_init"] = best_initialization
    for model_init in range(model_inits):
      new_model_entry[f"psnr_{model_init}"] = validation_psnrs[model_init]

    self.model_database.add(new_model_id, new_model_entry)

  def update_perf_database(self, model_id, cost, train_time):
    entry = {"model_id": model_id, "cost": cost, "train_time": train_time}
    self.perf_database.add(model_id, entry)


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


  """
  Builds valid input nodes for a chroma model given the model's designated green model choice
  """
  def construct_chroma_inputs(self, green_model_id):
    bayer = demosaic_ast.Input(4, "Bayer")
    green_GrGb = demosaic_ast.Input(2, "Green@GrGb")
    rb = demosaic_ast.Input(2, "RedBlueBayer")
    
    flat_green = demosaic_ast.Input(1, "GreenExtractor", no_grad=True, green_model_id=green_model_id)

    green_quad = demosaic_ast.Flat2Quad(flat_green)
    green_quad.compute_input_output_channels()
    green_quad_input = demosaic_ast.Input(4, "GreenQuad", node=green_quad, no_grad=True)

    green_rb = demosaic_ast.GreenRBExtractor(flat_green)
    green_rb.compute_input_output_channels()
    green_rb_input = demosaic_ast.Input(2, "Green@RB", node=green_rb, no_grad=True)

    rb_min_g = demosaic_ast.Sub(rb, green_rb)
    rb_min_g_stack_green = demosaic_ast.Stack(rb_min_g, green_quad)
    rb_min_g_stack_green.compute_input_output_channels()
    rb_min_g_stack_green_input = demosaic_ast.Input(6, "RBdiffG_GreenQuad", no_grad=True, node=rb_min_g_stack_green)

    return {
      "Input(Bayer)": bayer,
      "Input(Green@GrGb)": green_GrGb,
      "Input(GreenExtractor)": flat_green, # can select this as an input to insert, must pick GreenQuad instead
      "Input(GreenQuad)": green_quad_input,
      "Input(Green@RB)": green_rb_input,
      "Input(RBdiffG_GreenQuad)": rb_min_g_stack_green_input
    }
    
 
  """
  Builds valid input nodes for a green model
  """
  def construct_green_inputs(self):
    bayer = demosaic_ast.Input(4, "Bayer")

    return {
      "Input(Bayer)": bayer,
    }


  def mutate_model(self, parent_id, new_model_id, model_ast, generation, generation_stats, partner_ast=None):
    self.mutate_monitor.set_error_msg(f"---\nfailed to mutate model id {parent_id}\n---")
    with self.mutate_monitor:
      try:
        if self.args.full_model:
          green_model_id = get_green_model_id(model_ast)
          if partner_ast:
            set_green_model_id(partner_ast, green_model_id)
          model_inputs = self.construct_chroma_inputs(green_model_id)
          model_input_names = OrderedSet(model_inputs.keys())
          self.args.input_ops = OrderedSet([v for k,v in model_inputs.items() if k != "Input(GreenExtractor)"]) # green extractor is on flat bayer, can only use green quad input
        elif self.args.rgb8chan: # full rgb model search uses same inputs as green search
          model_inputs = self.construct_green_inputs()
          model_input_names = OrderedSet(model_inputs.keys())
          self.args.input_ops = OrderedSet(list(model_inputs.values()))
        else: 
          model_inputs = self.construct_green_inputs()
          model_input_names = OrderedSet(model_inputs.keys())
          self.args.input_ops = OrderedSet(list(model_inputs.values()))

        new_model_ast, shash, mutation_stats = self.mutator.mutate(
            parent_id, new_model_id, model_ast, model_input_names, generation,
            partner_ast=partner_ast)
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
              

  """
  inserts the green model ast referenced by the green_model_id stored in 
  an input node into the input node.
  NOTE: WE MAKE SURE ONLY ONE GREEN MODEL DAG IS CREATED SO THAT 
  IT IS ONLY RUN ONCE PER RUN OF THE FULL MODEL
  """
  def insert_green_model(self, new_model_ast, green_model=None, green_model_weight_file=None):
    if green_model is None:
      green_model_id = get_green_model_id(new_model_ast)
      green_model_ast_file = self.args.green_model_asts[green_model_id]
      green_model_weight_file = self.args.green_model_weights[green_model_id]

      green_model = demosaic_ast.load_ast(green_model_ast_file)

    nodes = new_model_ast.preorder()
    for n in nodes:
      if type(n) is demosaic_ast.Input:
        if n.name == "Input(GreenExtractor)":
          n.node = green_model
          n.weight_file = green_model_weight_file
        elif hasattr(n, "node"): # other input ops my run submodels that also require the green model
          self.insert_green_model(n.node, green_model, green_model_weight_file)


  def lower_model(self, new_model_id, new_model_ast):
    self.lowering_monitor.set_error_msg(f"---\nfailed to lower model {new_model_id}\n---")
    with self.lowering_monitor:
      try:
        new_models = [new_model_ast.ast_to_model() for i in range(self.args.model_initializations)]
      except TimeoutError:
        return None
      else:
        return new_models


  def create_worker(self, worker_id, validation_psnrs):
    gpu_id = worker_id
    worker = mp.Process(target=run_train, args=(self.work_queue, worker_id, self.pending_start_times, self.pending_tasks, self.pending_locks[worker_id], \
                                              self.finished_tasks, gpu_id, self.args, validation_psnrs, self.log_format, self.work_loggers[worker_id]))
    return worker


  def monitor_training_tasks(self, validation_psnrs):
    timeout = self.args.train_timeout
    if self.args.ram:
      bootup_time = 300
    else:
      bootup_time = 30
    
    alive = set()
    failed_tasks = []
    for wid, worker in enumerate(self.workers):
      worker.start()
      alive.add(wid)

    tick = 0
    while True:
      if tick % 3 == 0:
        self.work_manager_logger.info(f"alive workers: {alive}")
        print(f"work queue size {self.work_queue.qsize()}")

      if self.work_queue.empty():
        self.work_manager_logger.info("No more models in work queue, waiting for all tasks to complete")

      for wid, worker in enumerate(self.workers): 
        if not worker.is_alive() and (wid in alive):
          worker.join()
          alive.remove(wid)
          self.work_manager_logger.info(f"worker {wid} is dead with exit code {worker.exitcode}")

          pending_task = self.pending_tasks[wid]
          failed_tasks.append(pending_task)

        else: # check if worker has run out of time on current task
          self.pending_locks[wid].acquire()
          start_time = self.pending_start_times[wid]
          task_id = self.pending_tasks[wid]

          if start_time >= 0 and task_id >= 0:
            curr_time = time.time()

            terminated = False 
            if curr_time - start_time > bootup_time:
              if not os.path.exists(f"{self.training_tasks[task_id].model_dir}/v0_train_log"):
                self.work_manager_logger.info(f"worker {wid} running task {task_id} has no 'v0_train_log', terminating")
                worker.terminate()
                worker.join()
                terminated = True

            if curr_time - start_time > timeout:
              self.work_manager_logger.info(f"worker {wid} timed out, killing at {datetime.datetime.now()}")
              worker.terminate()
              worker.join()
              terminated = True

            if terminated:
              failed_tasks.append(task_id)
              new_worker = self.create_worker(wid, validation_psnrs)
              new_worker.start()
              self.workers[wid] = new_worker
          
          self.pending_locks[wid].release()
      
      if len(alive) == 0:
        assert self.work_queue.empty(), "all workers are dead with work left in the queue"
        self.work_manager_logger.info("All tasks are done")
        break

      time.sleep(10)
      tick += 1

    return failed_tasks


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


  def load_seed_models(self, cost_tiers):
    # load all seed models
    seed_model_id = self.model_manager.start_id
    seed_model_files = [l.strip() for l in open(self.args.seed_model_files)]
    seed_model_psnrs = [float(l.strip()) for l in open(self.args.seed_model_psnrs)]

    for seed_model_i, seed_model_file in enumerate(seed_model_files):
      seed_ast = demosaic_ast.load_ast(seed_model_file)
      seed_ast.compute_input_output_channels()
      if self.args.full_model:
        self.insert_green_model(seed_ast)

      seed_model_dir = self.model_manager.model_dir(seed_model_id)
      util.create_dir(seed_model_dir)
      seed_model = seed_ast.ast_to_model()
      self.model_manager.save_model([seed_model], seed_ast, seed_model_dir)

      compute_cost = self.evaluator.compute_cost(seed_ast)
      model_accuracy = seed_model_psnrs[seed_model_i]
      
      self.search_logger.info(f"seed model {seed_model_id}\ncost: {compute_cost}\npsnr: {model_accuracy}\n{seed_ast.dump()}")

      cost_tiers.add(seed_model_id, compute_cost, model_accuracy)

      if self.args.full_model:
        seed_green_model_id = get_green_model_id(seed_ast)
      else:
        seed_green_model_id = -1

      self.model_database.add(seed_model_id,\
              {'model_id': seed_model_id,
               'id_str' : seed_ast.id_string(),
               'hash': hash(seed_ast),
               'structural_hash': demosaic_ast.structural_hash(seed_ast),
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
               'seen_rejections': 0,
               'mutation_type': 'N/A',
               'insert_ops': 'N/A',
               'binop_operand_choice': 'N/A',
               'mutation_node_id': -1,
               'new_output_channels': -1,
               'new_grouping': -1,
               'green_model_id': seed_green_model_id})

      seed_model_id = self.model_manager.get_next_model_id()


  # searches over program mutations within tiers of computational cost
  def search(self, compute_cost_tiers, tier_size):
    cost_tiers = CostTiers(self.args.tier_database_dir, compute_cost_tiers, self.search_logger)

    if self.args.restart_generation is not None:
      if self.args.restart_tier == 0:
        cost_tiers.load_generation_from_database(self.args.tier_db_snapshot, self.args.restart_generation-1)
      else:
        cost_tiers.load_from_snapshot(self.args.tier_db_snapshot, self.args.tier_snapshot)
      self.model_database.load(self.args.model_db_snapshot)
      start_generation = self.args.restart_generation
      end_generation = start_generation + self.args.generations
    else:
      self.load_seed_models(cost_tiers)
      start_generation = 0
      end_generation = self.args.generations 

    self.search_logger.info(f"--- STARTING SEARCH AT GENERATION {start_generation} AND ENDING AT {end_generation} ---")
      
    for generation in range(start_generation, end_generation):
      generational_tier_sizes = [len(tier.items()) for tier in cost_tiers.tiers]
      if self.args.restart_generation and (generation == self.args.restart_generation):
        self.search_logger.info(f"--- STARTING GENERATION {generation} at tier {self.args.restart_tier} ---")
      else:
        self.search_logger.info(f"--- STARTING GENERATION {generation} ---")
      printstr = "tier sizes: "
      for t in generational_tier_sizes:
        printstr += f"{t} "
      self.search_logger.info(printstr)
      self.search_logger.info("---------------")

      generation_stats = GenerationStats()
    
      new_cost_tiers = copy.deepcopy(cost_tiers)

      task_id = 0
      self.training_tasks = {}
      validation_psnrs = mp.Array(ctypes.c_double, [-1]*self.models_per_gen)
      self.finished_tasks = mp.Array(ctypes.c_double, [-1]*self.models_per_gen)
      self.pending_start_times = mp.Array(ctypes.c_double, [-1]*self.num_workers)
      self.pending_tasks = mp.Array(ctypes.c_double, [-1]*self.num_workers)

      self.workers = [self.create_worker(worker_id, validation_psnrs) for worker_id in range(self.num_workers)]

      for tid, tier in enumerate(cost_tiers.tiers):
        if self.args.restart_generation and (generation == self.args.restart_generation) and (tid < self.args.restart_tier):
          continue
        if len(tier) == 0:
          continue

        if self.args.pareto_sampling:
          tier_sampler = pareto_util.Sampler(tier, self.args.pareto_factor)
        else:
          tier_sampler = cost.Sampler(tier)

        self.search_logger.info(f"\n--- sampling tier {tid} size: {len(tier)} min psnr: {tier_sampler.min} max psnr: {tier_sampler.max} ---")

        # importance sample which models from each tier to mutate based on PSNR
        model_ids = [tier_sampler.sample() for mutation in range(self.args.mutations_per_generation)]
        mutation_batch_info = MutationBatchInfo() # we'll store results from this mutation batch here          
       
        for model_id in model_ids:
          self.search_logger.info(f"loading model {model_id}")

          model_ast = self.load_model(model_id)
          if model_ast is None:
            continue

          if self.args.binop_change:
            other_model_ids = model_ids.copy()
            other_model_ids.remove(model_id)

            partner_model_id = random.sample(other_model_ids, 1)[0]
            partner_model_ast = self.load_model(partner_model_id)

          new_model_id = self.model_manager.get_next_model_id()
          if self.args.binop_change:
            new_model_ast, shash, mutation_stats = self.mutate_model(model_id, new_model_id, model_ast, generation, generation_stats, partner_ast=partner_model_ast)
          else:
            new_model_ast, shash, mutation_stats = self.mutate_model(model_id, new_model_id, model_ast, generation, generation_stats)

          if new_model_ast is None:
            continue

          print(f"model {new_model_id} hash: {hash(new_model_ast)}\n------------------")

          if self.args.full_model:
            self.insert_green_model(new_model_ast)

          pytorch_models = self.lower_model(new_model_id, new_model_ast)
          print(sys.getsizeof(pytorch_models[0])/1000)

          # green model must be inserted before computing cost if we're running full model search
          compute_cost = self.evaluator.compute_cost(new_model_ast)
          if compute_cost > new_cost_tiers.max_cost:
            self.debug_logger.info(f"dropping model with cost {compute_cost} - too computationally expensive")
            continue

          new_model_dir = self.model_manager.model_dir(new_model_id)
          new_model_entry = model_database_entry(new_model_id, new_model_ast.id_string(), new_model_ast, \
                                                shash, generation, compute_cost, model_id, mutation_stats)
          
          task_info = MutationTaskInfo(task_id, new_model_entry, new_model_dir, pytorch_models, new_model_id)
          subproc_task_info = {"task_id":task_id, "model_dir":new_model_dir, "models":pytorch_models, "model_id":new_model_id}

          util.create_dir(task_info.model_dir)
          self.save_model_ast(new_model_ast, task_info.model_id, task_info.model_dir)

          """
          seen_psnrs = mysql_db.find(self.args.mysql_auth, self.args.tablename, hash(new_model_ast), \
                                      new_model_ast.id_string(), self.args.experiment_name, self.mysql_logger)
          """
          seen_psnrs = None
          if not seen_psnrs is None: # model seen on other machine, skip training and use the given psnrs
            self.search_logger.info(f"model {new_model_id} already seen on another machine")
            self.update_model_database(task_info, seen_psnrs)
            best_psnr = max(seen_psnrs)
            if math.isnan(best_psnr) or best_psnr < 0:
              continue # don't add model to tier
            compute_cost = task_info.database_entry["compute_cost"]
            new_cost_tiers.add(task_info.model_id, compute_cost, best_psnr)
          else:
            print(f"PUTTING TASK {task_id}")
            self.work_queue.put(subproc_task_info)
            self.training_tasks[task_id] = task_info
            task_id += 1
        
      for w in range(self.num_workers):
        self.work_queue.put(None) # place sentinels telling workers to exit

      failed_tasks = self.monitor_training_tasks(validation_psnrs)

      print("finished tasks")
      # update model database 
      for task_id, task_time in enumerate(self.finished_tasks):
        if task_time < 0: 
          continue # this task was not run
        task_info = self.training_tasks[task_id]
        task_id = task_info.task_id 
        print(f"task_id {task_id} task id in task info: {task_info.task_id} model id {task_info.model_id}")

        index = task_id * self.args.keep_initializations
        model_psnrs = validation_psnrs[index:(index+self.args.keep_initializations)]
 
        if self.args.deterministic:
          model_psnrs = [random.uniform(0,1) * (33 - 28) + 28 for i in range(args.keep_initializations)]
        print(f"model {task_info.model_id} psnrs {model_psnrs}")

        # add model to cost tiers
        best_psnr = max(model_psnrs)
        if math.isnan(best_psnr) or best_psnr < 0:
          continue # don't add model to tier 

        self.search_logger.info(f"adding model {task_info.model_id} with psnrs {model_psnrs} to db")

        compute_cost = task_info.database_entry["compute_cost"]
        new_cost_tiers.add(task_info.model_id, compute_cost, best_psnr)
        
        self.update_model_database(task_info, model_psnrs)
        self.update_perf_database(task_info.model_id, compute_cost, task_time)
        """
        mysql_db.mysql_insert(self.args.mysql_auth, self.args.tablename, task_info.model_id, self.args.machine, \
                              self.args.save, self.args.experiment_name, hash(new_model_ast), new_model_ast.id_string(), model_psnrs, self.mysql_logger)
        """

      self.model_database.save()
      new_cost_tiers.save_snapshot(generation, tid)
        
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
  
    print(f"downsample\ntries: {self.mutator.binop_tries} loc select fails {self.mutator.binop_loc_selection_failures} \
        insert fail {self.mutator.binop_insertion_failures} reject {self.mutator.binop_rejects} success {self.mutator.binop_success} seen {self.mutator.binop_seen}")

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

  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--max_channels', type=int, default=32, help='max channel count')
  parser.add_argument('--default_channels', type=int, default=12, help='initial channel count for Convs')
  parser.add_argument('--max_nodes', type=int, default=40, help='max number of nodes in a tree')
  parser.add_argument('--min_subtree_size', type=int, default=1, help='minimum size of subtree in insertion')
  parser.add_argument('--max_subtree_size', type=int, default=15, help='maximum size of subtree in insertion')
  parser.add_argument('--structural_sim_reject', type=float, default=0.2, help='rejection probability threshold for structurally similar trees')
  parser.add_argument('--max_footprint', type=int, default=16, help='max DAG footprint size on input image')
  parser.add_argument('--crop', type=int, default=16, help='how much to crop images during training and inference')

  parser.add_argument('--starting_model_id', type=int)
  parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
  parser.add_argument('--model_database_dir', type=str, default='model_database', help='path to save model statistics')
  parser.add_argument('--failure_database_dir', type=str, default='failure_database', help='path to save mutation failure statistics')
  parser.add_argument('--performance_database_dir', type=str, default='performance_database', help='path to save system training time performance')
  parser.add_argument('--tier_database_dir', type=str, default='cost_tier_database', help='path to save cost tier snapshot')

  parser.add_argument('--restart_generation', type=int, help='generation to start search from if restarting a prior run')
  parser.add_argument('--restart_tier', type=int, help='tier to start search from if restarting a prior run')
  parser.add_argument('--tier_snapshot', type=str, help='saved cost tiers to restart from')
  parser.add_argument('--model_db_snapshot', type=str, help='saved model database to restart from')
  parser.add_argument('--tier_db_snapshot', type=str, help='saved tier database to restart from')

  parser.add_argument('--save', type=str, help='experiment name')

  parser.add_argument('--seed', type=int, default=1, help='random seed')
  parser.add_argument('--deterministic', action='store_true', help="set model psnrs to be deterministic")
  parser.add_argument('--ram', action='store_true')
  parser.add_argument('--lazyram', action='store_true')

  # seed models 
  parser.add_argument('--green_seed_model_files', type=str, help='file with list of filenames of green seed model asts')
  parser.add_argument('--green_seed_model_psnrs', type=str, help='file with list of psnrs of green seed models')

  parser.add_argument('--rgb8chan_seed_model_files', type=str, help='file with list of filenames of rgb8chan seed model asts')
  parser.add_argument('--rgb8chan_seed_model_psnrs', type=str, help='file with list of psnrs of rgb8chan seed models')

  parser.add_argument('--chroma_seed_model_files', type=str, help='file with list of filenames of chroma seed model asts')
  parser.add_argument('--chroma_seed_model_psnrs', type=str, help='file with list of psnrs of chroma seed models')
  
  parser.add_argument('--green_model_asts', type=str, help="file with list of filenames of green model asts")
  parser.add_argument('--green_model_weights', type=str, help="file with list of filenames of green model weights")

  parser.add_argument('--generations', type=int, default=20, help='model search generations')
  parser.add_argument('--cost_tiers', type=str, help='list of tuples of cost tier ranges')
  parser.add_argument('--tier_size', type=int, default=15, help='how many models to keep per tier')
  parser.add_argument('--pareto_sampling', action='store_true', help='whether to use pareto sampling')
  parser.add_argument('--pareto_factor', type=float, help='discount factor per pareto frontier')
  parser.add_argument('--mutations_per_generation', type=int, default=12, help='how many mutations produced by each tier per generation')

  parser.add_argument('--mutation_failure_threshold', type=int, default=100, help='max number of tries to mutate a tree')
  parser.add_argument('--delete_failure_threshold', type=int, default=25, help='max number of tries to find a node to delete')
  parser.add_argument('--subtree_selection_tries', type=int, default=50, help='max number of tries to find a subtree when inserting a binary op')
  parser.add_argument('--partner_insert_loc_tries', type=int, default=25, help='max number of tries to find a insert location for a partner op')
  parser.add_argument('--insert_location_tries', type=int, default=25, help='max number of tries to find an insert location for a chosen insert op')

  parser.add_argument('--load_timeout', type=int, default=10)
  parser.add_argument('--mutate_timeout', type=int, default=60)
  parser.add_argument('--lowering_timeout', type=int, default=10)
  parser.add_argument('--train_timeout', type=int, default=600)
  parser.add_argument('--save_timeout', type=int, default=10)

  # training parameters
  parser.add_argument('--num_gpus', type=int, default=4, help='number of available GPUs') # change this to use all available GPUs
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
  parser.add_argument('--lrsearch', action='store_true', help='whether or not to use lr search')
  parser.add_argument('--lr_search_steps', type=int, default=1, help='how many line search iters for finding learning rate')
  parser.add_argument('--variance_min', type=float, default=0.003, help='minimum validation psnr variance')
  parser.add_argument('--variance_max', type=float, default=0.02, help='maximum validation psnr variance')
  parser.add_argument('--weight_decay', type=float, default=1e-16, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=200, help='training report frequency')
  parser.add_argument('--save_freq', type=float, default=2000, help='trained weights save frequency')
  parser.add_argument('--epochs', type=int, default=3, help='num of training epochs')
  parser.add_argument('--model_initializations', type=int, default=3, help='number of initializations to train per model for the first epoch')
  parser.add_argument('--keep_initializations', type=int, default=1, help='how many initializations to keep per model after the first epoch')

  parser.add_argument('--train_portion', type=int, default=1e5, help='portion of training data to use')
  parser.add_argument('--training_file', type=str, default="/home/karima/cnn-data/sample_files.txt", help='filename of file with list of training data image files')
  parser.add_argument('--validation_file', type=str, default="/home/karima/cnn-data/sample_files.txt", help='filename of file with list of validation data image files')
  parser.add_argument('--validation_freq', type=int, default=50, help='validation frequency for assessing validation PSNR variance')
  parser.add_argument('--validation_variance_start_step', type=int, default=400, help='training step from which to start sampling validation PSNR for assessing variance')
  parser.add_argument('--validation_variance_end_step', type=int, default=1600, help='training step from which to start sampling validation PSNR for assessing variance')

  # training full chroma + green parameters
  parser.add_argument('--full_model', action="store_true")
  parser.add_argument('--rgb8chan', action="store_true")
  parser.add_argument('--binop_change', action="store_true")
  parser.add_argument('--insertion_bias', action="store_true")
  parser.add_argument('--demosaicnet_search', action="store_true", help='whether to run search over demosaicnet space')
  parser.add_argument('--late_cdf_gen', type=int, help='generation to switch to late mutation type cdf')
  
  parser.add_argument('--tablename', type=str)
  parser.add_argument('--mysql_auth', type=str)
  parser.add_argument("--machine", type=str)

  args = parser.parse_args()

  # if not torch.cuda.is_available():
  #   print("CUDA not found, aborting job.")
  #   sys.exit(1)

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  if args.full_model:
    args.seed_model_files = args.chroma_seed_model_files
    args.seed_model_psnrs = args.chroma_seed_model_psnrs
    args.green_model_asts = [l.strip() for l in open(args.green_model_asts)]
    args.green_model_weights = [l.strip() for l in open(args.green_model_weights)]
    args.task_out_c = 3
  elif args.rgb8chan:
    args.seed_model_files = args.rgb8chan_seed_model_files
    args.seed_model_psnrs = args.rgb8chan_seed_model_psnrs
    args.task_out_c = 3
  else:
    args.seed_model_files = args.green_seed_model_files
    args.seed_model_psnrs = args.green_seed_model_psnrs
    args.task_out_c = 1

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
