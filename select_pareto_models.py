import csv
import argparse
import os
import sys
from paretoset import paretoset
import pandas as pd 
import numpy as np
import math
import random
import cost
import demosaic_ast
import shutil

def parse_cost_tiers(s):
  ranges = s.split(' ')
  ranges = [[int(x) for x in r.split(',')] for r in ranges]
  return ranges

"""
assumes pareto info is sorted by cost
"""
def find_pareto_knee(pareto_info):
	print("tier info")
	print(pareto_info)
	cur_slope = None
	for i in range(len(pareto_info)-1):
		p1 = pareto_info[i]
		p2 = pareto_info[i+1]
		cost1 = p1[1]
		psnr1 = p1[2]
		cost2 = p2[1]
		psnr2 = p2[2]

		slope = (psnr2 - psnr1)/(cost2 - cost1) 
		if cur_slope and slope < cur_slope:
			print(f"knee {pareto_info[i-1]} {pareto_info[i]} {pareto_info[i+1]}")
			return pareto_info[i]
		cur_slope = slope
	# slope constantly increases
	return None



parser = argparse.ArgumentParser()
parser.add_argument('--cost_tiers', type=str, help='list of tuples of cost tier ranges')
parser.add_argument('--pareto_ids', type=str, help='file with model ids of pareto models from search results')
parser.add_argument('--retrained_pareto_dir', type=str, help='directory with retrained pareto model info')
parser.add_argument('--model_inits', type=int, default=3)
parser.add_argument('--seed_ast_file', type=str)
parser.add_argument('--seed_psnr_file', type=str)
parser.add_argument('--seed_weight_file', type=str)
parser.add_argument('--selected_models_dir', type=str, help='where to copy over selected models info')
parser.add_argument('--seed_files_dir', type=str)
args = parser.parse_args()


evaluator = cost.ModelEvaluator(None)

model_list = [l.strip() for l in open(args.pareto_ids, "r")]

model_ids = []
compute_costs = []
psnrs = []
best_versions = []

# load the retrained pareto model info if we have it
for model_id in model_list:
		model_dir = os.path.join(args.retrained_pareto_dir, f"{model_id}")
		model = demosaic_ast.load_ast(os.path.join(model_dir, "model_ast"))
		model_cost = evaluator.compute_cost(model)

		init_psnrs = []
		for init in range(args.model_inits):
			loss_file = os.path.join(model_dir, f"v{init}_validation_log")
			if not os.path.exists(loss_file):
				continue
			else:
				validation_psnrs = []
				for l in open(loss_file, "r"):
					retrained_psnr = float(l.split(' ')[-1].strip())
					validation_psnrs.append(retrained_psnr)

				if len(validation_psnrs) == 0:
					continue
				best_psnr = max(validation_psnrs)

			init_psnrs.append(best_psnr)

		if len(init_psnrs) == 0:
			continue

		model_ids += [model_id]
		compute_costs += [model_cost]
		model_psnr = max(init_psnrs)
		best_v = np.argmax(init_psnrs)
		best_versions += [best_v]
		print(f"model {model_id} cost: {model_cost} psnr: {model_psnr}")
		psnrs += [model_psnr]

data = pd.DataFrame(
	{
		"compute": compute_costs,
		"psnr": psnrs
	})

mask = paretoset(data, sense=["min", "max"])

pareto_info = [(model_ids[i], compute_costs[i], psnrs[i], best_versions[i]) for i, v in enumerate(mask) if v]
min_cost = min([m[1] for m in pareto_info])
max_cost = max([m[1] for m in pareto_info])
print(f"min cost {min_cost} max cost {max_cost}")

nbuckets = 10
base = math.sqrt(2)
factor = (max_cost-min_cost) / math.pow(base, nbuckets)
buckets = [min_cost + factor * math.pow(base, i) for i in range(0,nbuckets+1)]

print(len(pareto_info))
os.mkdir(args.seed_files_dir)
os.mkdir(args.selected_models_dir)

sorted_pareto = sorted(pareto_info, key=lambda x:x[2])

for i in range(1, len(buckets)):
	print(f"bucket {buckets[i-1], buckets[i]}")
	best_index = -1
	best_psnr = -1
	m = None
	for j, mm in enumerate(sorted_pareto):
		if mm[1] > buckets[i]:
			break
		if mm[2] > best_psnr and mm[1] <= buckets[i] and mm[1] > buckets[i-1]:
			best_psnr = mm[2]
			bset_index = j
			m = mm

	if m is None:
		continue
		
	print(f"best model in bucket: {m}")
	with open(os.path.join(args.seed_files_dir, args.seed_ast_file), "a+") as f:		
		f.write(os.path.join(args.selected_models_dir, f"{m[0]}/model_ast") + "\n")
	with open(os.path.join(args.seed_files_dir, args.seed_weight_file), "a+") as f:
		f.write(os.path.join(args.selected_models_dir, f"{m[0]}/model_v{m[3]}_pytorch") + "\n")
	with open(os.path.join(args.seed_files_dir, args.seed_psnr_file), "a+") as f:
		f.write(f"{m[2]}" + "\n")

	model_dir = os.path.join(args.selected_models_dir, f"{m[0]}")
	print(model_dir)
	shutil.copytree(os.path.join(args.retrained_pareto_dir, f"{m[0]}"), model_dir)



# cost_tiers = parse_cost_tiers(args.cost_tiers)
# for tier in cost_tiers:
# 	within_tier = [t for t in pareto_info if t[1] <tier[1] and t[1]>=tier[0]]
# 	within_tier = sorted(within_tier, key=lambda x:x[2])

# 	print(f"tier {tier}")

# 	if len(within_tier) > 1:
# 		model_1 = within_tier[0]
# 		model_2 = within_tier[-1]
# 		knee_model = find_pareto_knee(within_tier)
# 		if knee_model:
# 			chosen_models = [model_1, knee_model, model_2]
# 		else:
# 			chosen_models = [model_1, model_2]
# 	elif len(within_tier) > 0:
# 		chosen_models = [within_tier[0]]
# 	else: # no pareto models in this tier
# 		continue
# 	print(chosen_models)
# 	for m in chosen_models:
# 		with open(os.path.join(args.seed_files_dir, args.seed_ast_file), "a+") as f:		
# 			f.write(os.path.join(args.selected_models_dir, f"{m[0]}/model_ast") + "\n")
# 		with open(os.path.join(args.seed_files_dir, args.seed_weight_file), "a+") as f:
# 			f.write(os.path.join(args.selected_models_dir, f"{m[0]}/model_v{m[3]}_pytorch") + "\n")
# 		with open(os.path.join(args.seed_files_dir, args.seed_psnr_file), "a+") as f:
# 			f.write(f"{m[2]}" + "\n")

# 		model_dir = os.path.join(args.selected_models_dir, f"{m[0]}")
# 		print(model_dir)
# 		shutil.copytree(os.path.join(args.retrained_pareto_dir, f"{m[0]}"), model_dir)

