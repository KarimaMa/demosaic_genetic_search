import csv
import argparse
import os
import sys
from paretoset import paretoset
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import math
import random


def parse_cost_tiers(s):
  ranges = s.split(' ')
  ranges = [[int(x) for x in r.split(',')] for r in ranges]
  print(ranges)
  return ranges


def get_tier(cost_tiers, compute_cost):
	for tid, tier in enumerate(cost_tiers):
		if compute_cost < tier[1]:
			return tid

def count_inversions(psnrs):
	inversions = 0
	for i in range(len(psnrs)):
		inversions += sum(other_psnr < psnr for other_psnr in psnrs[i+1:]) 
	return inversions

parser = argparse.ArgumentParser()
parser.add_argument("--sampled_models_csv", type=str, help='file with sampled model info')
parser.add_argument("--search_models_dir", type=str, help='directory with search models')
parser.add_argument("--retrained_models_dir", type=str, help='directory with retrained models')
parser.add_argument("--model_inits", type=int, default=3)
parser.add_argument("--num_tiers", type=str, help='number of cost tiers')
args = parser.parse_args()

sampled_models = {}

with open(args.sampled_models_csv,  newline='\n') as csvf:
	reader = csv.DictReader(csvf, delimiter=',')
	for r in reader:
		sampled_models[r['model_id']] = {'cost': r["compute_cost"], 'psnr': r["psnr"], tier: r['tier']}

for model_id, model_info in sampled_models.items():
	model_dir = os.path.join(args.search_models_dir, f"{model_id}")
	init_psnrs = []
	for init in range(args.model_inits):
		loss_file = os.path.join(model_dir, f"v{init}_validation_log")
		if not os.path.exists(loss_file):
			continue
		else:
			validation_psnrs = []
			for l in open(loss_file, "r"):
				loss = float(l.split(' ')[-1].strip())
				retrain_psnr = compute_psnr(loss)
				validation_psnrs.append(retrain_psnr)

			if len(validation_psnrs) == 0:
				continue
			best_psnr = max(validation_psnrs)

		init_psnrs.append(best_psnr)

	if len(init_psnrs) == 0:
		continue
	new_psnr = max(init_psnrs)
	print(f"model {model_id} search db psnr: {sampled_models[model_id]["psnr"]} stored psnr {new_psnr}")
	sampled_models[model_id]["psnr"] = new_psnr	


for model_id, model_info in sampled_models.items():
	model_dir = os.path.join(args.retrained_models_dir, f"{model_id}")
	init_psnrs = []
	for init in range(args.model_inits):
		loss_file = os.path.join(model_dir, f"v{init}_validation_log")
		if not os.path.exists(loss_file):
			continue
		else:
			validation_psnrs = []
			for l in open(loss_file, "r"):
				loss = float(l.split(' ')[-1].strip())
				retrain_psnr = compute_psnr(loss)
				validation_psnrs.append(retrain_psnr)

			if len(validation_psnrs) == 0:
				continue
			best_psnr = max(validation_psnrs)

		init_psnrs.append(best_psnr)

	if len(init_psnrs) == 0:
		continue
	new_psnr = max(init_psnrs)
	print(f"model {model_id} new psnr: {new_psnr}")
	sampled_models[model_id]["retrained_psnr"] = new_psnr	

tier_psnrs = {}
retrained_tier_psnrs = {}

for tid in range(args.num_tiers):
	tier_psnrs[tid] = []
	retrained_tier_psnrs[tid] = []

for model_id, model_info in sampled_models.items():
	search_psnr = model_info["psnr"]
	retrained_psnr = model_info["retrained_psnr"]
	tier_psnrs[model_info["tier"]] += [search_psnr]
	retrained_tier_psnrs[model_info["tier"]] += [retrained_psnr]

for tid in range(args.num_tiers):
	search_psnrs = tier_psnrs[tid]
	retrained_psnrs = retrained_tier_psnrs[tid]
	retrained_psnrs_sorted = [x for _,x in sorted(zip(search_psnrs, retrained_psnrs))]
	print(search_psnrs)
	print(retrained_psnrs_sorted)






