import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils
from dataset import GreenDataset


class KcoreSet():
  def __init__(self, initial_features, initial_ids):
    self.ids = set(initial_ids.tolist())
    self.features = initial_features.clone().detach() # pytorch tensor
  
  def size(self):
    return len(self.ids)

  def add(self, feature, sample_id):
    if sample_id in self.ids:
      return
    self.ids.add(sample_id.item()) 
    self.features = torch.cat([self.features, feature.clone().detach()], dim=0)

"""
computes minimum dist between features and kcore_set
"""
def compute_kcore_dist(features, kcore_set, n):
  kcore_dists = torch.zeros(n)
  for i in range(n):
    dists = torch.norm(kcore_set.features - features[i], dim=1)
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

  while True:
    candidate_id = None
    candidate_feature = None
    candidate_dist = 0.0

    for step, (sample_ids, input, _) in enumerate(train_queue):       
      # use the first batch as the seed subset
      kcore_features = {}
      n = input.size(0)
      input = Variable(input, requires_grad=False).cuda()
      _ = model.run(input, kcore_features)
      if step == 0:
        kcore_set = KcoreSet(kcore_features["feat"], sample_ids)
        continue
      
      dists = compute_kcore_dist(kcore_features["feat"], kcore_set, n)
      max_dist = torch.max(dists)
      if max_dist > candidate_dist:
        max_id = torch.argmax(dists)
        candidate_id = sample_ids[max_id]
        candidate_dist = max_dist
        candidate_feature = kcore_features["feat"][max_id].unsqueeze(0)

      # can't afford to run through the whole dataset just to get one sample
      # so we add to the kcore set every some predetermined interval
      if (step+1) % (len(train_queue) // args.samples_per_iter) == 0:
        kcore_set.add(candidate_feature, candidate_id)
        # reset 
        candidate_id = None
        candidate_feature = None
        candidate_dist = 0.0

      if step % args.report_freq == 0:
        logger.info('adding elem %03d batch %03d', kcore_set.size(), step)
        kcore_ids = list(kcore_set.ids)
        kcore_ids.sort()
        train_data.save_kcore_filenames(kcore_ids, args.output_file)

      if kcore_set.size() >= args.budget:
        break 
    if kcore_set.size() >= args.budget:
      break 

  kcore_ids = list(kcore_set.ids)
  kcore_ids.sort()
  train_data.save_kcore_filenames(kcore_ids, args.output_file)
  return kcore_set

