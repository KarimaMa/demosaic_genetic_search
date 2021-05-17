import sys
import os
import math

rootdir = '/'.join(sys.path[0].split("/")[0:-1])
sys.path.append(rootdir)

import cost
import demosaic_ast


def get_best_model_version(training_info_file):
	with open(training_info_file, "r") as f:
		for l in f:
			if l.find("keeping best model") >= 0:
				return int(l.split(" ")[-1])
	return None

def get_model_psnr(model_dir, training_info_file):
	best_model_version = get_best_model_version(training_info_file)
	if best_model_version is None:
		print(model_dir)
	else:
		validation_psnr_file = os.path.join(model_dir, f"v{best_model_version}_validation_log")
		val_psnrs = [float(l.split(" ")[-1]) for l in open(validation_psnr_file, "r")]
		max_psnr = max(val_psnrs)
		return max_psnr, best_model_version

	return None, None

def get_model_cost(model_ast_file):
	evaluator = cost.ModelEvaluator(None)
	model = demosaic_ast.load_ast(model_ast_file)
	return evaluator.compute_cost(model)


def parse_cost_tiers(s):
	ranges = s.split(' ')
	ranges = [[int(x) for x in r.split(',')] for r in ranges]
	return ranges

def collect_model_info(modeldir, max_id):
	psnrs = []
	costs = []
	model_ids = []
	best_inits = []

	evaluator = cost.ModelEvaluator(None)

	for model in os.listdir(modeldir):
		model_id = int(model)
		if model_id > max_id:
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

	return model_ids, costs, psnrs, best_inits

