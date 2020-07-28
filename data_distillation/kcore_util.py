import torch
import torch.nn as nn
import torch.utils
from dataset import GreenDataset


class KcoreSet():
	def __init__(self, initial_features, initial_ids):
		self.ids = set(initial_ids.tolist())
		self.features = initial_features # pytorch tensor
	
	def size(self):
		return len(self.ids)

	def add(self, feature, sample_id):
		if sample_id in self.ids:
			return
		self.ids.add(sample_id) 
		self.features = torch.cat([self.features, feature], dim=0)

"""
computes minimum dist between features and kcore_set
"""
def compute_kcore_dist(features, kcore_set, n):
	kcore_dists = np.zeros(n)
	for i in range(n):
		dists = torch.norm(kcore_set.features - features[i], 2)
		min_dist = torch.min(dists)
		kcore_dists[i] = min_dist
	return kcore_dists

def kcore_greedy(model, args, logger):
	model.eval()
	train_data = GreenDataset(args.training_file, return_index=True) 

	num_train = len(train_data)
	train_indices = list(range(num_train))

	train_queue = torch.utils.data.DataLoader(
			train_data, batch_size=args.batch_size,
			sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
			pin_memory=True, num_workers=0)

	while kcore_set.size() < args.budget:
		candidate_id = None
		candidate_feature = None
		candidate_dist = 0.0

		for step, (sample_ids, input, _) in enumerate(train_queue):				
			# use the first batch as the seed subset
			if step == 0
				kcore_set = KcoreSet(input, sample_ids)
				continue

			kcore_features = None
			n = input.size(0)
			input = Variable(input, requires_grad=False).cuda()
			_ = model.run(input)
			
			dists = compute_kcore_dist(kcore_features, kcore_set, n)
			max_dist = torch.max(dists)

			if max_dist > candidate_dist:
				max_id = torch.argmax(dists)
				candidate_id = sample_ids[max_id]
				candidate_dist = max_dist
				candidate_feature = kcore_features[max_id]

			# can't afford to run through the whole dataset just to get one sample
			# so we add to the kcore set every some predetermined interval
			# but note that we keep increasing the distance threshold as we go through
			# one iteration of the training set
			if (step+1) % (len(train_queue) // args.samples_per_iter) == 0:
				kcore_set.add(candidate_feature, candidate_id)

			if step % args.report_freq == 0:
				logger.info('adding elem %03d batch %03d', elem, step)
			
	return kcore_set

			

