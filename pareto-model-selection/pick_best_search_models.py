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
import shutil


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--retrained_models", type=str, help="directory of retrained pareto models")
	parser.add_argument("--search_models", type=str, help="directory of pareto models trained during search")

	parser.add_argument("--pareto_info_csv", type=str, help="csv file with pareto model ids, (pre-retraining) psnrs, and costs")
	parser.add_argument("--n", type=int, help="how many pareto models to select")
	parser.add_argument("--outdir", type=str, help="where to copy selected model info")

	args = parser.parse_args()

	retrained_model_info = {"modelid": [], "psnr":[], "cost":[]}

	with open(args.pareto_info_csv, "r") as f:
		reader = csv.DictReader(f)
		for r in reader:
			psnr = float(r["psnr"])
			cost = float(r["cost"])
			model_id = r["modelid"]

			model_dir = os.path.join(args.retrained_models, model_id)
			training_info_file = os.path.join(model_dir, f"model_{model_id}_training_log")
			retrained_psnr, best_version, best_epoch = get_model_psnr(model_dir, training_info_file, return_epoch=True)
			if best_epoch == 0:
				print(f"using epoch 0 {model_dir}")
			if retrained_psnr == 0:
				print(f"model {model_dir} has all inf psnr")

			if retrained_psnr < psnr:
				retrained_psnr = psnr

			retrained_model_info["modelid"].append(model_id)
			retrained_model_info["psnr"].append(retrained_psnr)
			retrained_model_info["cost"].append(cost)


	model_ids = retrained_model_info["modelid"]
	costs = retrained_model_info["cost"]
	psnrs = retrained_model_info["psnr"]

	min_cost = min([c for c in costs])
	max_cost = max([c for c in costs])

	model_perfs = zip(model_ids, costs, psnrs)

	print(f"min cost {min_cost} max cost {max_cost}")
	print(f"min psnr {min(psnrs)} max psnr {max(psnrs)}")

	picked_ids = []
	picked_psnrs = []
	picked_costs = []
	
	nbuckets = 10
	base = math.sqrt(2)
	factor = (max_cost-min_cost) / math.pow(base, nbuckets)
	buckets = [min_cost + factor * math.pow(base, i) for i in range(0,nbuckets+1)]

	if not os.path.exists(args.outdir):
		os.mkdir(args.outdir)

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
		
		print(f"best model in bucket {i} : {m}")
		model_id = m[0]
		model_psnr = m[2]
		model_cost = m[1]

		model_dir = os.path.join(args.retrained_models, model_id)
		selected_model_dir = os.path.join(args.outdir, model_id)

		shutil.copytree(model_dir, selected_model_dir)

		picked_ids.append(model_id)
		picked_costs.append(model_cost)
		picked_psnrs.append(model_psnr)

	with open(os.path.join(args.outdir, "picked_model_psnrs_costs.csv"), "w") as f:
		writer = csv.DictWriter(f, fieldnames=["modelid", "cost", "psnr"])
		writer.writeheader()
		for i in range(len(picked_ids)):
			writer.writerow({"modelid": picked_ids[i], "cost": picked_costs[i], "psnr":picked_psnrs[i]})




