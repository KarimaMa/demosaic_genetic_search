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


def compute_psnr(loss):
  return 10*math.log(math.pow(255,2) / math.pow(math.sqrt(loss)*255, 2),10)

def parse_cost_tiers(s):
  ranges = s.split(' ')
  ranges = [[int(x) for x in r.split(',')] for r in ranges]
  print(ranges)
  return ranges


def get_tier(cost_tiers, compute_cost):
	for tid, tier in enumerate(cost_tiers):
		if compute_cost < tier[1]:
			return tid

def count_inversions(psnrs, threshold, model_ids, model_info):
	inversions = 0
	inversion_indices = []
	inversion_counts = []
	for i in range(len(psnrs)):
		model_inversions = sum(other_psnr < (psnrs[i]-threshold) for other_psnr in psnrs[i+1:]) 
		inversions += model_inversions
		if model_inversions > 0:	
			inversion_counts += [model_inversions]
			inversion_indices += [i]
	for i in range(len(inversion_indices)):
		index = inversion_indices[i]
		print(f"model {model_ids[index]}  inversions {inversion_counts[i]}")
		thispsnr = psnrs[index]
		for j, otherpsnr in enumerate(psnrs):
			if j > index and otherpsnr < thispsnr - threshold:
				otherid = model_ids[j]
				thisid = model_ids[index]
				print(f"model {thisid} cost: {model_info[thisid]['cost']}  og: {model_info[thisid]['psnr']:.2f}  psnr {thispsnr:.2f} model {otherid}  cost: {model_info[otherid]['cost']}  og: {model_info[otherid]['psnr']:.2f}  psnr {otherpsnr:.2f} ") 
			
					
	return inversions

parser = argparse.ArgumentParser()
parser.add_argument("--sampled_models_csv", type=str, help='file with sampled model info')
parser.add_argument("--search_models_dir", type=str, help='directory with search models')
parser.add_argument("--retrained_models_dir", type=str, help='directory with retrained models')
parser.add_argument("--model_inits", type=int, default=3)
parser.add_argument("--num_tiers", type=int, help='number of cost tiers')
args = parser.parse_args()

sampled_models = {}

with open(args.sampled_models_csv,  newline='\n') as csvf:
	reader = csv.DictReader(csvf, delimiter=',')
	for r in reader:
		sampled_models[r['model_id']] = {'cost': r["compute_cost"], 'psnr': float(r["psnr"]), 'tier': int(r['tier'])}

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
	print(f"model {model_id} search db psnr: {sampled_models[model_id]['psnr']} stored psnr {new_psnr}")
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
	print(f"model {model_id}  tier {sampled_models[model_id]['tier']}   cost {sampled_models[model_id]['cost']}  search psnr: {sampled_models[model_id]['psnr']:.2f}  new psnr: {new_psnr:.2f}")
	sampled_models[model_id]["retrained_psnr"] = new_psnr	

tier_psnrs = {}
retrained_tier_psnrs = {}

for tid in range(args.num_tiers):
	tier_psnrs[tid] = {"model_ids": [], "psnrs": []}
	retrained_tier_psnrs[tid] = {"model_ids": [], "psnrs": []}


for model_id, model_info in sampled_models.items():
	search_psnr = model_info["psnr"]
	if not "retrained_psnr" in model_info:
		continue

	retrained_psnr = model_info["retrained_psnr"]
	tier_psnrs[model_info["tier"]]["model_ids"] += [model_id]
	tier_psnrs[model_info["tier"]]["psnrs"] += [search_psnr]
	retrained_tier_psnrs[model_info["tier"]]["model_ids"] += [model_id]
	retrained_tier_psnrs[model_info["tier"]]["psnrs"] += [retrained_psnr]

for tid in range(args.num_tiers):
	search_psnrs = tier_psnrs[tid]["psnrs"]
	retrained_psnrs = retrained_tier_psnrs[tid]["psnrs"]
	search_model_ids = tier_psnrs[tid]["model_ids"]
	retrained_model_ids = retrained_tier_psnrs[tid]["model_ids"]
	num_models = len(search_model_ids)
	max_inversions = num_models * (num_models - 1)/2

	sorted_by_search_psnrs = sorted(zip(search_psnrs, retrained_psnrs, search_model_ids))
	sorted_ids = [x for _,_,x in sorted_by_search_psnrs]

	retrained_psnrs_sorted = [x for _,x,_ in sorted_by_search_psnrs]
	inversions = count_inversions(retrained_psnrs_sorted, 0.2, sorted_ids, sampled_models)	
	print(f"tier: {tid} inversions {inversions} out of {max_inversions} possible inversions  {inversions/max_inversions}%")
	print("------")






