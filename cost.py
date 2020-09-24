"""
Computes per pixel cost of a program tree
"""

import util
import operator
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from demosaic_ast import *
from dataset import GreenDataset

ADD_COST = 1
MUL_COST = 1
DIV_COST = 10
LOGEXP_COST = 10
RELU_COST = 1
DOWNSAMPLE_FACTOR_SQ = 9
DIRECTIONS = 4
KERNEL_W = 5


class CostTiers():
	def __init__(self, compute_cost_ranges):
		self.tiers = [{} for r in compute_cost_ranges]
		self.compute_cost_ranges = compute_cost_ranges
		self.max_cost = compute_cost_ranges[-1][1]
	"""
	model_file is file with model topology and model weights
	"""
	def add(self, model_id, compute_cost, model_accuracy):
		for i, cost_range in enumerate(self.compute_cost_ranges):
			if compute_cost < cost_range[1]:
				self.tiers[i][model_id] = (compute_cost, model_accuracy)
				return
		assert False, f"model cost {compute_cost} exceeds max tier cost range"

	"""
	keeps the top k performing models in terms of model accuracy per tier 
	"""
	def keep_topk(self, k):
		for tid, tier in enumerate(self.tiers):
			if len(tier) == 0:
				continue
			sorted_models = sorted(tier.items(), key= lambda item: item[1][1])
			new_tier = {}

			for i in range(min(k, len(sorted_models))):
				new_tier[sorted_models[i][0]] = sorted_models[i][1]
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
		cost = 0
		if isinstance(root, Input):
			return cost
		elif isinstance(root, Add) or isinstance(root, Sub):
			cost += root.in_c[0] * ADD_COST
			cost += self.compute_cost(root.lchild)
			cost += self.compute_cost(root.rchild)
		elif isinstance(root, Mul):
			cost += root.in_c[0] * MUL_COST
			cost += self.compute_cost(root.lchild)
			cost += self.compute_cost(root.rchild) 
		elif isinstance(root, LogSub) or isinstance(root, AddExp):
			cost += root.in_c[0] * (2*LOGEXP_COST + ADD_COST)
			cost += self.compute_cost(root.lchild)
			cost += self.compute_cost(root.rchild) 
		elif isinstance(root, Stack):
			cost += self.compute_cost(root.lchild)
			cost += self.compute_cost(root.rchild) 
		elif isinstance(root, ChromaExtractor) or isinstance(root, GreenExtractor):
			cost += self.compute_cost(root.lchild)
			cost += self.compute_cost(root.rchild) 
		elif isinstance(root, Softmax):
			cost += root.in_c * (LOGEXP_COST + DIV_COST + ADD_COST)
			cost += self.compute_cost(root.child)
		elif isinstance(root, Relu):
			cost += root.in_c * RELU_COST
			cost += self.compute_cost(root.child)
		elif isinstance(root, Log) or isinstance(root, Exp):
			cost += root.in_c * LOGEXP_COST	
			cost += self.compute_cost(root.child)
		elif isinstance(root, Downsample):
			cost += root.in_c * ADD_COST 
			cost += self.compute_cost(root.child) 
			cost *= DOWNSAMPLE_FACTOR_SQ
		elif isinstance(root, Upsample):
			cost += root.in_c * ADD_COST
			cost += self.compute_cost(root.child) / DOWNSAMPLE_FACTOR_SQ
		elif isinstance(root, Conv1x1):
			cost += root.in_c * root.out_c * MUL_COST
			cost += self.compute_cost(root.child)
		elif isinstance(root, Conv1D):
			cost += root.in_c * root.out_c * DIRECTIONS * KERNEL_W * MUL_COST
			cost += self.compute_cost(root.child)
		elif isinstance(root, Conv2D):
			cost += root.in_c * root.out_c * KERNEL_W * KERNEL_W * MUL_COST
			cost += self.compute_cost(root.child)
		elif isinstance(root, SumR):
			cost += root.in_c * ADD_COST
			cost += self.compute_cost(root.child)
		else:
			assert False, "compute cost encountered unexpected node type"
		return cost

