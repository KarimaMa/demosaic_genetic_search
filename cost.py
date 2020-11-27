"""
Computes per pixel cost of a program tree
"""
import random
import util
import operator
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np 

from demosaic_ast import *
from dataset import GreenDataset
from database import Database
from pareto_util import get_pareto_ranks

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
	loads everything from stored snapshot up to (not including) the end generation
	"""
	def load_generation_from_database(self, database_file, generation):
		self.logger.info(f"--- Reloading cost tiers from {database_file} generation {generation} ---")
		self.build_cost_tier_database()
		self.tier_database.load(database_file)
		
		self.tier_database.cntr = max(self.tier_database.table.keys())

		for key, data in self.tier_database.table.items():
			if data["generation"] == generation:
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
		
	"""
	trains and evaluates model
	"""
	def performance_cost(models, model_id, model_dir):
		train_loggers = [util.create_logger(f'{model_id}_v{i}_train_logger', logging.INFO, \
																			self.log_format, os.path.join(model_dir, f'v{i}_train_log'))\
										for i in range(len(models))]

		# validation_logger = [util.create_logger(f'{model_id}_validation_logger', logging.INFO, \
		# 																	self.log_format, os.path.join(model_dir, f'v{i}_validation_log')) \
		# 										for i in range(len(models))]

		model_pytorch_files = [util.get_model_pytorch_file(model_dir, model_version) \
													for model_version in range(len(models))]

		models = [model.cuda() for model in models]

		criterion = nn.MSELoss()

		optimizers = [torch.optim.SGD(
				model.parameters(),
				self.args.learning_rate,
				momentum=self.args.momentum,
				weight_decay=self.args.weight_decay) for m in models]

		train_data = GreenDataset(self.args.training_file) 
		validation_data = GreenDataset(self.args.validation_file)

		num_train = len(train_data)
		train_indices = list(range(num_train*args.train_portion))
		num_validation = len(validation_data)
		validation_indices = list(range(num_validation))

		train_queue = torch.utils.data.DataLoader(
				train_data, batch_size=self.args.batch_size,
				sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
				pin_memory=True, num_workers=0)

		validation_queue = torch.utils.data.DataLoader(
				validation_data, batch_size=self.args.batch_size,
				sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices),
				pin_memory=True, num_workers=0)

		for epoch in range(args.epochs):
			# training
			train_losses = train(train_queue, models, criterion, optimizers, train_loggers, model_pytorch_files)
			# validation
			valid_losses = infer(validation_queue, models, criterion)

		return valid_obj

	def train(train_queue, models, criterion, optimizers, train_loggers, model_pytorch_files):
		loss_trackers = [utils.AvgrageMeter() for m in models]
		for m in models:
			m.train()

		for step, (input, target) in enumerate(train_queue):
			n = input.size(0)

			input = Variable(input, requires_grad=False).cuda()
			target = Variable(target, requires_grad=False).cuda()

			for i, model in enumerate(models):
				optimizers[i].zero_grad()
				pred = model(input)
				loss = criterion(pred, target)

				loss.backward()
				optimizers[i].step()

				loss_trackers[i].update(loss.data[0], n)

			if step % self.args.report_freq == 0:
				for i in range(len(models)):
					train_loggers[i].info('train %03d %e', step, loss_trackers[i].avg)

			if step % self.args.save_freq == 0:
				for i in range(len(models)):
					torch.save(models[i], model_pytorch_files[i])

		return [loss_tracker.avg for loss_tracker in loss_trackers]


	def infer(valid_queue, models, criterion):
		loss_trackers = [utils.AvgrageMeter() for m in models]
		for m in models:
			model.eval()

		for step, (input, target) in enumerate(valid_queue):
			input = Variable(input, volatile=True).cuda()
			target = Variable(target, volatile=True).cuda()
			n = input.size(0)

			for i, model in enumerate(models):
				pred = model(input)
				loss = criterion(pred, target)

				loss_trackers[i].update(loss.data[0], n)

		return [loss_tracker.avg for loss_tracker in loss_trackers]


	def compute_cost(self, root):
		return self.compute_cost_helper(root) / 4 
		
	def compute_cost_helper(self, root):
		cost = 0
		if isinstance(root, Input):
			return cost
		elif isinstance(root, Add) or isinstance(root, Sub):
			cost += root.in_c[0] * ADD_COST
			cost += self.compute_cost_helper(root.lchild)
			cost += self.compute_cost_helper(root.rchild)
		elif isinstance(root, Mul):
			cost += root.in_c[0] * MUL_COST
			cost += self.compute_cost_helper(root.lchild)
			cost += self.compute_cost_helper(root.rchild) 
		elif isinstance(root, LogSub) or isinstance(root, AddExp):
			cost += root.in_c[0] * (2*LOGEXP_COST + ADD_COST)
			cost += self.compute_cost_helper(root.lchild)
			cost += self.compute_cost_helper(root.rchild) 
		elif isinstance(root, Stack):
			cost += self.compute_cost_helper(root.lchild)
			cost += self.compute_cost_helper(root.rchild) 
		elif isinstance(root, RGBExtractor):
			cost += self.compute_cost_helper(root.child1)
			cost += self.compute_cost_helper(root.child2)
			cost += self.compute_cost_helper(root.child3)
		elif isinstance(root, GreenExtractor):
			cost += self.compute_cost_helper(root.lchild)
			cost += self.compute_cost_helper(root.rchild) 
		elif isinstance(root, Softmax):
			cost += root.in_c * (LOGEXP_COST + DIV_COST + ADD_COST)
			cost += self.compute_cost_helper(root.child)
		elif isinstance(root, Relu):
			cost += root.in_c * RELU_COST
			cost += self.compute_cost_helper(root.child)
		elif isinstance(root, Log) or isinstance(root, Exp):
			cost += root.in_c * LOGEXP_COST	
			cost += self.compute_cost_helper(root.child)
		elif isinstance(root, Downsample):
			downsample_k = SCALE_FACTOR * 2
			cost += root.in_c * root.out_c * downsample_k * downsample_k * MUL_COST
			cost += self.compute_cost_helper(root.child) 
		elif isinstance(root, Upsample):
			cost += root.in_c * BILINEAR_COST
			cost += self.compute_cost_helper(root.child) / (SCALE_FACTOR**2)
		elif isinstance(root, Conv1x1):
			cost += root.groups * ((root.in_c // root.groups) * (root.out_c // root.groups) * MUL_COST)
			cost += self.compute_cost_helper(root.child)
		elif isinstance(root, Conv1D):
			cost += root.groups * ((root.in_c // root.groups) * (root.out_c // root.groups) * DIRECTIONS * root.kwidth * MUL_COST)
			cost += self.compute_cost_helper(root.child)
		elif isinstance(root, Conv2D):
			cost += root.groups * ((root.in_c // root.groups) * (root.out_c // root.groups) * root.kwidth**2 * MUL_COST)
			cost += self.compute_cost_helper(root.child)
		elif isinstance(root, InterleavedSum) or isinstance(root, GroupedSum):
			cost += ((root.in_c / root.out_c) - 1) * root.out_c * ADD_COST
			cost += self.compute_cost_helper(root.child)
		else:
			print(type(root))
			assert False, "compute cost encountered unexpected node type"

		return cost

