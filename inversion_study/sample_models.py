import csv
import argparse
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


def write_pareto_model_csv(out_csvfile, model_ids_outfile, sampled_models):
	sampled_model_ids = []
	with open(out_csvfile, 'w', newline='\n') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=['model_id', 'compute_cost', 'psnr', 'generation', 'tier'])
		writer.writeheader()
		for tier, tier_samples in sampled_models.items():
			for idx, model_id in enumerate(tier_samples["model_ids"]):
				data = {'model_id': model_id, 'compute_cost': tier_samples["costs"][idx], 
								'psnr': tier_samples["psnrs"][idx], "tier": tier}
				writer.writerow(data)
				sampled_model_ids.append(model_id)

	with open(model_ids_outfile, "w+") as f:
		for model_id in sampled_model_ids:
			f.write(f"{model_id}\n")



parser = argparse.ArgumentParser()
parser.add_argument("--csvfile", type=str, help='model database csv')
parser.add_argument("--cost_tiers", type=str, help='cost tiers')
parser.add_argument("--samples", type=int, help="number of samples to take per tier")
parser.add_argument("--sample_csvfile", type=str, help="csv file to write sample info")
parser.add_argument("--sample_model_ids_files", type=str, help="file to write sampled model ids")
args = parser.parse_args()

cost_tiers = parse_cost_tiers(args.cost_tiers)


models_by_tier = {}
for tid, tier in enumerate(cost_tiers):
	models_by_tier[tid] = {"model_ids": [], "costs": [], "psnrs": [], "generations": []}

with open(args.csvfile, newline='\n') as csvf:
	reader = csv.DictReader(csvf, delimiter=',')
	for r in reader:	
		best_psnr = max([float(p) for p in [r["psnr_0"], r["psnr_1"], r["psnr_2"]]])
		compute_cost = int(float(r["compute_cost"]))
		tier = get_tier(cost_tiers, compute_cost)

		models_by_tier[tier]["model_ids"].append(r["model_id"])
		models_by_tier[tier]["costs"].append(compute_cost)
		models_by_tier[tier]["psnrs"].append(best_psnr)
		models_by_tier[tier]["generations"].append(r["generation"])


sampled_models = {}

# sample models from each tier
for tid, tier_info in models_by_tier.items():
	sampled_indices = random.sample([idx for idx in range(len(tier_info["model_ids"]))], args.samples)
	model_ids = np.array(tier_info["model_ids"])[sampled_indices]
	psnrs = np.array(tier_info["psnrs"])[sampled_indices]
	costs = np.array(tier_info["costs"])[sampled_indices]
	generations = np.array(tier_info["generations"])[sampled_indices]
	sampled_models[tid] = {"model_ids": model_ids, "costs": costs, "psnrs": psnrs, "generations": generations}

write_pareto_model_csv(args.sample_csvfile, args.sample_model_ids_files, sampled_models)



