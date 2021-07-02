import sys
import os
import math
import numpy as np 

rootdir = '/'.join(sys.path[0].split("/")[0:-1])
sys.path.append(rootdir)
multinode_sys_run_dir = os.path.join(rootdir, "multinode_sys_run")
sys.path.append(multinode_sys_run_dir)

import cost
import demosaic_ast
from search_util import insert_green_model



def get_best_model_version(training_info_file):
	# if os.path.exists(training_info_file):
	# 	with open(training_info_file, "r") as f:
	# 		for l in f:
	# 			if l.find("keeping best model") >= 0:
	# 				return int(l.split(" ")[-1])
	# no models were dropped, look through all training logs
	train_dir = os.path.dirname(training_info_file)
	inits = 3
	epoch0_psnrs = []
	validation_logs = [os.path.join(train_dir, f"v{i}_validation_log") for i in range(inits)]
	for log in validation_logs:
		psnrs = []	
		for l in open(log, "r"):
			if "epoch" in l:
				psnrs.append(float(l.split(" ")[-1].strip()))
		for i,p in enumerate(psnrs):
			if math.isinf(p):
				psnrs[i] = 0

		if len(psnrs) == 0:
			epoch0_psnrs.append(-1)
		else:
			epoch0_psnrs.append(psnrs[0])

	return np.argmax(epoch0_psnrs)


def get_validation_psnrs(validation_log):
	val_psnrs = []
	with open(validation_log, "r") as f:
		for l in f:
			if "epoch" in l:
				val_psnrs.append(float(l.split(" ")[-1]))
	return val_psnrs


def get_model_psnr(model_dir, training_info_file, return_epoch=False):
	best_model_version = get_best_model_version(training_info_file)
	print(f"best version {best_model_version}")
	if best_model_version is None:
		print(model_dir)
	else:
		validation_psnr_file = os.path.join(model_dir, f"v{best_model_version}_validation_log")
		val_psnrs = get_validation_psnrs(validation_psnr_file)
		for i, p in enumerate(val_psnrs):
			if math.isinf(p):
				val_psnrs[i] = 0

		if len(val_psnrs) == 0:
			print(f"{model_dir} no psnrs")
			max_psnr = 0
			best_epoch = -1
		else:
			max_psnr = max(val_psnrs)
			best_epoch = np.argmax(val_psnrs)
		if return_epoch:
			return max_psnr, best_model_version, best_epoch
		return max_psnr, best_model_version

	return None, None

def get_model_cost(model_ast_file, green_model_ast_files=None, green_model_weight_files=None):
	evaluator = cost.ModelEvaluator(None)
	model = demosaic_ast.load_ast(model_ast_file)

	if not green_model_ast_files is None:
		insert_green_model(model, green_model_ast_files, green_model_weight_files)

	return evaluator.compute_cost(model)


def parse_cost_tiers(s):
	ranges = s.split(' ')
	ranges = [[int(x) for x in r.split(',')] for r in ranges]
	return ranges

def is_in_range(model_id, id_ranges):
	return any([model_id <= r[1] and model_id >= r[0] for r in id_ranges])

def collect_model_info(modeldir, id_ranges, green_model_ast_files=None, green_model_weight_files=None):
	psnrs = []
	costs = []
	model_ids = []
	best_inits = []

	evaluator = cost.ModelEvaluator(None)

	skipped = 0
	for model in os.listdir(modeldir):
		model_id = int(model)
		if not id_ranges is None:
			if not is_in_range(model_id, id_ranges):
				skipped += 1
				continue

		model_dir = os.path.join(modeldir, model)

		training_info_file = os.path.join(model_dir, f"model_{model}_training_log")
		model_ast_file = os.path.join(model_dir, "model_ast")

		if not os.path.exists(training_info_file):
			continue
		if not os.path.exists(model_ast_file):
			continue

		model_psnr, init = get_model_psnr(model_dir, training_info_file)
		model_cost = get_model_cost(model_ast_file)

		if model_psnr is None or math.isinf(model_psnr) or model_cost is None:
			continue

		model_ids.append(model)
		psnrs.append(model_psnr)
		costs.append(model_cost)
		best_inits.append(init)

	print(f"skipped {skipped} models out of id ranges")
	return model_ids, costs, psnrs, best_inits

