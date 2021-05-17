import argparse
import os
import sys
from paretoset import paretoset
import pandas as pd 
import numpy as np
import shutil
from data_scrape_util import *

import csv

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--search_models", type=str, help="directories of search models")
	parser.add_argument("--num_buckets", type=int, help="how many buckets to pick best models from")
	parser.add_argument("--infodir", type=str)
	parser.add_argument("--ast_file", type=str, default="ast_files.txt", help="where to write paths to selected models asts")
	parser.add_argument("--weight_file", type=str, default="weight_files.txt", help="where to write paths to selected models weights")
	parser.add_argument("--psnr_file", type=str, default="psnrs.txt", help="where to write paths to selected models psnrs")
	parser.add_argument("--cost_file", type=str, default="costs.txt", help="where to write paths to selected models costs")
	parser.add_argument("--ids_file", type=str, default="modelids.txt", help="where to write paths to selected model ids")
	parser.add_argument("--max_id", type=str, default=-1)
	parser.add_argument("--selected_models", type=str, default="models", help="where to copy over model data for selected models")

	args = parser.parse_args()

	model_ids, costs, psnrs, best_inits = collect_model_info(args.search_models, args.max_id)

	# take the best model in each cost tier
	min_cost = min([c for c in costs])
	max_cost = max([c for c in costs])

	model_perfs = zip(model_ids, costs, psnrs, best_inits)

	print(f"min cost {min_cost} max cost {max_cost}")
	print(max(psnrs))
	
	nbuckets = 10
	base = math.sqrt(2)
	factor = (max_cost-min_cost) / math.pow(base, nbuckets)
	buckets = [min_cost + factor * math.pow(base, i) for i in range(0,nbuckets+1)]

	os.mkdir(args.infodir)
	selected_models_rootdir = os.path.join(args.infodir, args.selected_models)
	os.mkdir(selected_models_rootdir)

	sorted_models = sorted(model_perfs, key=lambda x:x[1])

	for i in range(1, len(buckets)):
		print(f"bucket {buckets[i-1], buckets[i]}")
		best_psnr = -1
		m = None
		for j, mm in enumerate(sorted_models):
			if mm[1] > buckets[i]:
				break
			if mm[2] > best_psnr and mm[1] <= buckets[i] and mm[1] > buckets[i-1]:
				best_psnr = mm[2]
				m = mm

		if m is None:
			continue
		
		print(f"best model in bucket: {m}")
		model_id = m[0]
		model_psnr = m[2]
		best_init = m[3]
		model_cost = m[1]

		model_dir = os.path.join(args.search_models, model_id)
		selected_model_dir = os.path.join(selected_models_rootdir, model_id)

		with open(os.path.join(args.infodir, args.ast_file), "a+") as f:		
			f.write(os.path.join(selected_model_dir, "model_ast") + "\n")

		with open(os.path.join(args.infodir, args.weight_file), "a+") as f:
			f.write(os.path.join(selected_model_dir, f"model_weights") + "\n")

		with open(os.path.join(args.infodir, args.psnr_file), "a+") as f:
			f.write(f"{model_psnr}\n")

		with open(os.path.join(args.infodir, args.cost_file), "a+") as f:
			f.write(f"{model_cost}\n")

		with open(os.path.join(args.infodir, args.ids_file), "a+") as f:
			f.write(f"{model_id}\n")

		shutil.copytree(model_dir, selected_model_dir)


