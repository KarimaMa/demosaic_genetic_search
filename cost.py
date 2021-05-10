"""
Computes per pixel cost of a program tree
"""
import random
import operator
import logging
import argparse
import pickle
import numpy as np 
import time
import os
import math

from demosaic_ast import *
from pareto_util import get_pareto_ranks
from database import Database

ADD_COST = 1
MUL_COST = 1
DIV_COST = 10
LOGEXP_COST = 10
RELU_COST = 1
DOWNSAMPLE_FACTOR_SQ = 4
DIRECTIONS = 2
KERNEL_W = 3

SCALE_FACTOR = 2
BILINEAR_COST = 3

"""
probabilistically picks models to reproduce according to their PSNR
"""
class Sampler():
  def __init__(self, cost_tier):
    self.cost_tier = cost_tier
    self.model_pdf = {}
    self.min = 50
    self.max = 0
    self.base = min([perf for _, (_, perf) in self.cost_tier.items()])
    self.perf_sum = self.base

    for model_id, (compute_cost, model_perf) in self.cost_tier.items():
      self.max = max(self.max, model_perf)
      self.min = min(self.min, model_perf)
      self.model_pdf[model_id] = (self.perf_sum, self.perf_sum + (model_perf-self.base))
      self.perf_sum += (model_perf - self.base)

  def sample(self):
    value = random.uniform(self.base, self.perf_sum)
    for model_id, (perf_min, perf_max) in self.model_pdf.items():
      if value <= perf_max and value >= perf_min:
        return model_id


class CostTiers():
  def __init__(self, database_dir, compute_cost_ranges, logger):
    self.database_dir = database_dir
    self.tiers = [{} for r in compute_cost_ranges]
    self.compute_cost_ranges = compute_cost_ranges
    self.max_cost = compute_cost_ranges[-1][1]
    self.logger = logger
    self.build_cost_tier_database()

  def build_cost_tier_database(self):
    fields = ["model_id", "generation", "tier", "compute_cost", "psnr"]
    field_types = [int, int, int, float, float]

    tier_database = Database("TierDatabase", fields, field_types, self.database_dir)
    self.tier_database = tier_database
    self.tier_database.cntr = 0

  def update_database(self, generation):
    for tid, tier in enumerate(self.tiers):
      for model_id in tier:
        compute_cost, psnr = tier[model_id]
        data = {"model_id" : model_id, 
          "generation" : generation, 
          "tier" : tid, 
          "compute_cost" : compute_cost, 
          "psnr" : psnr}
        self.tier_database.add(self.tier_database.cntr, data)
        self.tier_database.cntr += 1

  """
  Keeps track of which models are in each tier PRIOR to pareto filtering
  at the end of the generation so that we can restart the search in the middle of a generation
  """
  def save_snapshot(self, generation, tier):
    self.logger.info(f"--- saving tier snapshot for generation {generation} after completeing tier {tier} ---")
    snapshot_file = 'gen-{}-snapshot-{}'.format(generation, tier, time.strftime("%Y%m%d-%H%M%S"))
    snapshot_path = os.path.join(self.database_dir, snapshot_file)

    with open(snapshot_path, "wb") as f:      
      pickle.dump(self.tiers, f)


  """
  loads everything from stored snapshot 
  """
  def load_from_snapshot(self, database_file, snapshot_file):
    self.logger.info(f'--- loading cost tier database from {database_file} ---')
    self.build_cost_tier_database()
    self.tier_database.load(database_file)
    
    self.tier_database.cntr = max(self.tier_database.table.keys())
  
    self.logger.info(f"--- Reloading cost tiers from {snapshot_file} ---")
    with open(snapshot_file, "rb") as f:
      self.tiers = pickle.load(f)
  
    for tid, tier in enumerate(self.tiers):
      for model_id in tier:
        self.logger.info(f"tier {tid} : model {model_id} compute cost {tier[model_id][0]}, psnr {tier[model_id][1]}")

  """
  loads everything from stored snapshot up to (not including) the end generation
  """
  def load_generation_from_database(self, database_file, generation):
    self.logger.info(f"--- Reloading cost tiers from {database_file} generation {generation} ---")
    self.build_cost_tier_database()
    self.tier_database.load(database_file)
    
    self.tier_database.cntr = max(self.tier_database.table.keys())

    for key, data in self.tier_database.table.items():
      if data["generation"] == generation:
        if not math.isinf(data["psnr"]):                            
          self.tiers[data["tier"]][data["model_id"]] = (data["compute_cost"], data["psnr"])

    for tid, tier in enumerate(self.tiers):
      for model_id in tier:
        self.logger.info(f"tier {tid} : model {model_id} compute cost {tier[model_id][0]}, psnr {tier[model_id][1]}")

  """
  loads everything from stored snapshot up to (not including) the end generation
  """
  def load_from_database(self, database_file, end_generation):
    self.logger.info(f"--- Reloading cost tiers from {database_file} up to generation {end_generation} ---")
    self.build_cost_tier_database()
    self.tier_database.load(database_file)
    # remove any entries with generation >= end_generation
    to_delete = [key for (key, data) in self.tier_database.table.items() if data["generation"] >= end_generation]
    for key in to_delete:
      del self.tier_database.table[key]

    self.tier_database.cntr = len(self.tier_database.table)

    for key, data in self.tier_database.table.items():
      self.tiers[data["tier"]][data["model_id"]] = (data["compute_cost"], data["psnr"])

    for tid, tier in enumerate(self.tiers):
      for model_id in tier:
        self.logger.info(f"tier {tid} : model {model_id} compute cost {tier[model_id][0]}, psnr {tier[model_id][1]}")

  """
  model_file is file with model topology and model weights
  """
  def add(self, model_id, compute_cost, model_accuracy):
    for i, cost_range in enumerate(self.compute_cost_ranges):
      if compute_cost <= cost_range[1]:
        self.tiers[i][model_id] = (compute_cost, model_accuracy)        
        self.logger.info(f"adding model {model_id} with compute cost " +
                f"{compute_cost} and psnr {model_accuracy} to tier {i}")
        return
    assert False, f"model cost {compute_cost} exceeds max tier cost range"

  """
  keeps the top k performing models in terms of model accuracy per tier 
  """
  def keep_topk(self, k):
    for tid, tier in enumerate(self.tiers):
      if len(tier) == 0:
        continue
      sorted_models = sorted(tier.items(), key= lambda item: item[1][1], reverse=True)
      new_tier = {}

      for i in range(min(k, len(sorted_models))):
        new_tier[sorted_models[i][0]] = sorted_models[i][1]
      self.tiers[tid] = new_tier


  def pareto_keep_topk(self, k):
    self.logger.info(f"--- Pareto Top K culling ---")
    for tid, tier in enumerate(self.tiers):
      if len(tier) == 0:
        continue
      model_ids = list(tier.keys())
      values = list(zip(*[tier[model_id] for model_id in model_ids]))
      compute_costs = values[0]
      psnrs = values[1]
      ranks = get_pareto_ranks(compute_costs, psnrs)

      new_tier = {}
      curr_rank = 0
      while len(new_tier) < k and curr_rank <= max(ranks):
        frontier_indices = np.argwhere(ranks == curr_rank).flatten()
        rank_size = len(frontier_indices)
        if len(new_tier) + rank_size <= k:
          for f_idx in frontier_indices:
            new_tier[model_ids[f_idx]] = (compute_costs[f_idx], psnrs[f_idx])
            self.logger.info(f"tier {tid} adding model with rank {ranks[f_idx]} cost {compute_costs[f_idx]} psnr {psnrs[f_idx]}")
        else:
          portion = k - len(new_tier)
          chosen = np.random.choice(frontier_indices, portion)
          for f_idx in chosen:
            new_tier[model_ids[f_idx]] = (compute_costs[f_idx], psnrs[f_idx])
            self.logger.info(f"tier {tid} adding model with rank {ranks[f_idx]} cost {compute_costs[f_idx]} psnr {psnrs[f_idx]}")
        curr_rank += 1

      self.tiers[tid] = new_tier

class ModelEvaluator():
  def __init__(self, training_args):
    self.args = training_args
    self.log_format = '%(asctime)s %(levelname)s %(message)s'

  def compute_cost(self, root):
    return self.compute_cost_helper(root, set())
    
  def compute_cost_helper(self, root, seen):
    cost = 0
    if id(root) in seen:
      return cost
    else:
      seen.add(id(root))

    if isinstance(root, Input):
      if hasattr(root, "node"):
        if not id(root.node) in seen:
          cost += self.compute_cost_helper(root.node, seen)
          seen.add(id(root.node))
    elif isinstance(root, Add) or isinstance(root, Sub):
      cost += root.in_c[0] * ADD_COST
      cost += self.compute_cost_helper(root.lchild, seen)
      cost += self.compute_cost_helper(root.rchild, seen)
    elif isinstance(root, Mul):
      cost += root.in_c[0] * MUL_COST
      cost += self.compute_cost_helper(root.lchild, seen)
      cost += self.compute_cost_helper(root.rchild, seen) 
    elif isinstance(root, LogSub) or isinstance(root, AddExp):
      cost += root.in_c[0] * (2*LOGEXP_COST + ADD_COST)
      cost += self.compute_cost_helper(root.lchild, seen)
      cost += self.compute_cost_helper(root.rchild, seen) 
    elif isinstance(root, Stack):
      cost += self.compute_cost_helper(root.lchild, seen)
      cost += self.compute_cost_helper(root.rchild, seen) 
    elif isinstance(root, RGBExtractor):
      cost += self.compute_cost_helper(root.child1, seen)
      cost += self.compute_cost_helper(root.child2, seen)
      cost += self.compute_cost_helper(root.child3, seen)
      ratio = 1/4 
      cost *= ratio
    elif isinstance(root, RGB8ChanExtractor):
      cost += self.compute_cost_helper(root.lchild, seen)
      cost += self.compute_cost_helper(root.rchild, seen) 
      ratio = 1/4
      cost *= ratio
    elif isinstance(root, FlatRGB8ChanExtractor):
      cost += self.compute_cost_helper(root.lchild, seen)
      cost += self.compute_cost_helper(root.rchild, seen) 
    elif isinstance(root, GreenExtractor):
      cost += self.compute_cost_helper(root.lchild, seen)
      cost += self.compute_cost_helper(root.rchild, seen) 
      ratio = 1/4
      cost *= ratio
    elif isinstance(root, SGreenExtractor):
      cost += self.compute_cost_helper(root.child, seen)
    elif isinstance(root, XGreenExtractor):
      cost += self.compute_cost_helper(root.lchild, seen)
      cost += self.compute_cost_helper(root.rchild, seen)
      ratio = 1/36
      cost *= ratio
    elif isinstance(root, XFlatGreenExtractor):
      cost += self.compute_cost_helper(root.lchild, seen)
      cost += self.compute_cost_helper(root.rchild, seen)    
      ratio = 16/36
      cost *= ratio
    elif isinstance(root, XFlatRGBExtractor):
      cost += self.compute_cost_helper(root.child1, seen)
      cost += self.compute_cost_helper(root.child2, seen)
      cost += self.compute_cost_helper(root.child3, seen)
      ratio = 28/36
      cost *= ratio
    elif isinstance(root, GreenRBExtractor):
      cost += self.compute_cost_helper(root.child, seen)
    elif isinstance(root, XGreenRBExtractor):
      cost += self.compute_cost_helper(root.child, seen)
    elif isinstance(root, Softmax):
      cost += root.in_c * (LOGEXP_COST + DIV_COST + ADD_COST)
      cost += self.compute_cost_helper(root.child, seen)
    elif isinstance(root, Relu):
      cost += root.in_c * RELU_COST
      cost += self.compute_cost_helper(root.child, seen)
    elif isinstance(root, Log) or isinstance(root, Exp):
      cost += root.in_c * LOGEXP_COST 
      cost += self.compute_cost_helper(root.child, seen)
    elif isinstance(root, LearnedDownsample):
      downsample_k = root.factor * 2
      cost += root.in_c * root.out_c * downsample_k**2 * MUL_COST
      cost += (root.factor**2) * self.compute_cost_helper(root.child, seen) 
    elif isinstance(root, Pack):
      cost += (root.factor**2) * self.compute_cost_helper(root.child, seen) 
    elif isinstance(root, BilinearUpsample):
      cost += root.in_c * BILINEAR_COST
      cost += self.compute_cost_helper(root.child, seen) / (SCALE_FACTOR**2)
    elif isinstance(root, LearnedUpsample):
      cost += root.groups * ((root.in_c // root.groups)) * (root.out_c // root.groups) * root.factor**2
      cost += self.compute_cost_helper(root.child, seen) / (root.factor**2)
    elif isinstance(root, Unpack):
      cost += self.compute_cost_helper(root.child, seen) / (root.factor**2)
    elif isinstance(root, Conv1x1):
      cost += root.groups * ((root.in_c // root.groups) * (root.out_c // root.groups) * MUL_COST)
      cost += self.compute_cost_helper(root.child, seen)
    elif isinstance(root, Conv1D):
      cost += root.groups * ((root.in_c // root.groups) * (root.out_c // root.groups) * DIRECTIONS * root.kwidth * MUL_COST)
      cost += self.compute_cost_helper(root.child, seen)
    elif isinstance(root, Conv2D):
      cost += root.groups * ((root.in_c // root.groups) * (root.out_c // root.groups) * root.kwidth**2 * MUL_COST)
      cost += self.compute_cost_helper(root.child, seen)
    elif isinstance(root, InterleavedSum) or isinstance(root, GroupedSum):
      cost += ((root.in_c / root.out_c) - 1) * root.out_c * ADD_COST
      cost += self.compute_cost_helper(root.child, seen)
    elif isinstance(root, Flat2Quad):
      cost += self.compute_cost_helper(root.child, seen)
    else:
      assert False, f"compute cost encountered unexpected node type {type(root)}"

    return cost

