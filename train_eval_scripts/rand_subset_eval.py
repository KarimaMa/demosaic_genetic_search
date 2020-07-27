import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import logging
import argparse
import glob
import os
import sys
import random 
import numpy as np

import util
from torch_model import ast_to_model
from dataset import GreenDataset
from train_seed_model import infer

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--save', type=str, default='RAND_DATA_SUBSET_EVAL', help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--model_path', type=str, help='directory where subset models are stored')
  parser.add_argument('--subsets', type=int, help='the number of subset models')
  parser.add_argument('--training_file', type=str, help='a file with list of training data images')
  parser.add_argument('--results_file', type=str, default='eval_results', help='where to store evaluation results')
  parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')


  args = parser.parse_args()
  util.create_dir(args.save, scripts_to_save=glob.glob('*.py'))
  args.results_file = os.path.join(args.save, args.results_file)

  log_format = '%(asctime)s %(levelname)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

  experiment_logger = util.create_logger('experiment_logger', logging.INFO, log_format, \
                                os.path.join(args.save, 'experiment_log'))
  experiment_logger.info("args = %s", args)

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = False
  torch.manual_seed(args.seed)

  cudnn.enabled=True
  cudnn.deterministic=True
  torch.cuda.manual_seed(args.seed)

  if not torch.cuda.is_available():
    sys.exit(1)

  model_manager = util.ModelManager(args.model_path)

  criterion = nn.MSELoss()
  train_data = GreenDataset(args.training_file)
  num_train = len(train_data)
  train_indices = list(range(int(num_train)))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=True, num_workers=0)


  for subset in range(args.subsets):
    print(f"evaluating subset {subset} models")
    model_id = f'subset_{subset}'
    model_dir = model_manager.model_dir(model_id)

    with open(os.path.join(model_dir, "model_info")) as f:
      lines = len([l for l in f])
      model_versions = lines-1

    infer_loggers = [util.create_logger(f'{model_id}_v{version}_inference_logger', logging.INFO, log_format, \
                                        os.path.join(args.save, f'{model_id}_v{version}_inference_log'))\
                    for version in range(model_versions)]
    models = []
    for version in range(model_versions):
      model, ast = model_manager.load_model(model_id, version)
      models.append(model)

    full_training_losses = infer(args, train_queue, [models], criterion, [infer_loggers])

    with open(args.results_file, "a+") as f:
      full_training_losses = [str(tl) for tl in full_training_losses]
      full_training_losses_str = ",".join(full_training_losses)
      data_string = f"subset {subset} full training losses: {full_training_losses_str} \n"
      f.write(data_string)




