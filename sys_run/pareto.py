from paretoset import paretoset
import pandas as pd 
import math
import random

def get_pareto_ranks(self, compute_costs, psnrs):
	data = pd.DataFrame({
			"compute": compute_costs,
			"psnr": psnrs
		})

	ranks = np.zeros(len(data))
	remaining = data
	frontier_count = 0
	frontier_mask = np.array([True for i in range(len(data))])
	while len(remaining) > 0:
		mask = paretoset(remaining, sense=["min", "max"])
		frontier = remaining[mask]
		for index, row in data.iterrows():
			if ((frontier['compute'] == row['compute']) & (frontier['psnr'] == row['psnr'])).any():
				ranks[index] = frontier_count
		remaining = remaining[~mask]
		frontier_count += 1

	return ranks


def get_pareto_pdf(ranks, factor):
	unnormalized_weights = [1.0/math.pow(factor, r) for r in ranks]
	norm = sum(unnormalized_weights)
	weights = [uw / norm for uw  in unnormalized_weights]
	return weights


class Sampler():
	def __init__(self, cost_tier, factor):
		self.cost_tier = cost_tier
		self.factor = factor

		self.model_ids = self.cost_tier.keys()
		values = zip(*[self.cost_tier[model_id] for model_id in model_ids])
		self.compute_costs = values[0]
		self.psnrs = values[1]
		self.ranks = get_pareto_ranks(self.compute_costs, self.psnrs)
		self.pdf = get_pareto_pdf(self.ranks, self.factor)
		self.cdf = np.cumsum(self.pdf)

	def sample(self):
		value = random.uniform(0,1)
		for i in range(len(self.cdf)):
			if value < self.cdf[i]:
				return self.model_ids[i]




