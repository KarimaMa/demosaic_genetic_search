import argparse
import os
import sys
from paretoset import paretoset
import pandas as pd 
import numpy as np
from data_scrape_util import *
import math

import csv
import random


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--search_models", type=str, help="directory of search models")
	parser.add_argument('--green_model_asts', type=str, help="file with list of filenames of green model asts")	
	parser.add_argument('--green_model_weights', type=str, help="file with list of filenames of green model weights")
	parser.add_argument('--full_model', action='store_true')
	parser.add_argument("--outdir", type=str)
	parser.add_argument("--pareto_csv", type=str, default="pareto_model_psnrs_costs.csv", help="where to write out pareto models")
	parser.add_argument("--pareto_model_ids", type=str, default="model_ids.txt", help="where to write pareto model ids")
	parser.add_argument("--all_csv", type=str, default="all_model_psnrs_costs.csv")
	parser.add_argument("--n", type=int, help="how many pareto models to select")
	parser.add_argument("--min_psnr", type=float, default=0)
	parser.add_argument("--id_ranges", type=str)

	args = parser.parse_args()

	if args.id_ranges:
		id_ranges = [(int(x.split(",")[0]), int(x.split(",")[1])) for x in args.id_ranges.split(" ")]
		print(f"valid id ranges {id_ranges}")
	else:
		id_ranges = None

	if args.full_model:
		args.green_model_ast_files = [l.strip() for l in open(args.green_model_asts)]
		args.green_model_weight_files = [l.strip() for l in open(args.green_model_weights)]

	psnrs = []
	costs = []
	model_ids = []

	evaluator = cost.ModelEvaluator(None)

	count = 0
	skipped = 0
	for model in os.listdir(args.search_models):
		model_dir = os.path.join(args.search_models, model)
		training_info_file = os.path.join(model_dir, f"model_{model}_training_log")
		model_ast_file = os.path.join(model_dir, "model_ast")
		
		model_id = int(model)
		if not id_ranges is None:
			if not is_in_range(model_id, id_ranges):
				skipped += 1
				continue

		if not os.path.exists(training_info_file):
			continue
		if not os.path.exists(model_ast_file):
			continue

		if count % 100 == 0:
			print(count)
		count += 1
		
		model_psnr, best_version = get_model_psnr(model_dir, training_info_file)
		if args.full_model:
			model_cost = get_model_cost(model_ast_file, args.green_model_ast_files, args.green_model_weight_files)
		else:
			model_cost = get_model_cost(model_ast_file)

		if model_psnr is None or math.isinf(model_psnr) or model_cost is None:
			continue
		if model_psnr < args.min_psnr:
			continue
			
		model_ids.append(model)
		psnrs.append(model_psnr)
		costs.append(model_cost)

	print(f"skipped: {skipped} models... using {count} models")
	data = pd.DataFrame(
		{
			"compute": costs,
			"psnr": psnrs
		})

	tranches = np.ones(len(data)) * -1
	remaining = data 

	# extract pareto frontiers
	tranche = 0
	while len(remaining) > (len(model_ids) - args.n):
		print(f"full data len {len(data)} len remaining {len(remaining)}")
		mask = paretoset(remaining, sense=["min", "max"])
		frontier = remaining[mask]
		for index, row in data.iterrows():
			if ((frontier['compute'] == row['compute']) & (frontier['psnr'] == row['psnr'])).any():
				tranches[index] = tranche

		remaining = remaining[~mask]
		print(f"remaining {len(remaining)} {len(model_ids) - args.n}")
		tranche += 1

	tranches = tranches.astype(np.int)

	rank_counts = []
	for r in range(max(tranches)+1):
		count = np.count_nonzero(tranches == r)
		print(f"rank {r} count {count}")
		rank_counts.append(count)
	rank_csum = np.cumsum(rank_counts)

	max_rank = max(tranches)
	print(f"max rank {max_rank}")

	pareto_ids = [(i, model_ids[i]) for i,r in enumerate(tranches) if (r >= 0 and r < max_rank)]

	if rank_csum[max_rank] > args.n:
		# randomly sample remaining models needed
		remainder = args.n - rank_csum[max_rank-1]
		max_rank_models = [(i, model_ids[i]) for i,r in enumerate(tranches) if r == max_rank]
		sampled = random.sample(max_rank_models, remainder)
		pareto_ids += sampled
		print(f"remainder {remainder}")

	print(len(pareto_ids))
	print(tranches)

	if not os.path.exists(args.outdir):
		os.mkdir(args.outdir)
		
	# figure out which models to pick as pareto

	with open(os.path.join(args.outdir, args.pareto_csv), "w") as f:
		writer = csv.DictWriter(f, fieldnames=["modelid", "cost", "psnr", "rank"])
		writer.writeheader()
		for i, model_id in pareto_ids:
			writer.writerow({"modelid": model_id, "cost":costs[i], "psnr":psnrs[i]})

	with open(os.path.join(args.outdir, args.all_csv), "w") as f:
		writer = csv.DictWriter(f, fieldnames=["modelid", "cost", "psnr", "rank"])
		writer.writeheader()
		for i in range(len(model_ids)):
			writer.writerow({"modelid": model_ids[i], "cost":costs[i], "psnr":psnrs[i]})

	with open(os.path.join(args.outdir, args.pareto_model_ids), "w") as f:
		for x in pareto_ids:
			f.write(f"{x[1]}\n")






