import logging
import torch
import os
from functools import reduce

from demosaic_ast import load_ast
import shutil
import torch_model
import math
import csv
import logging



# from a github gist by victorlei
def extclass(cls):
  return lambda f: (setattr(cls,f.__name__,f) or f)


def get_csv_writer(filename):
  if os.path.exists(filename):
    os.remove(filename)
  csv_f = open(filename, 'w', newline='\n')
  writer = csv.writer(csv_f, delimiter=',')
  return writer

def compute_psnr(loss):
  return 10*math.log(float(255)**2 / math.pow(math.sqrt(loss)*float(255), 2),10)

def get_factors(n):    
  factors = set(reduce(list.__add__, 
    ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
  return factors

"""
finds the two factors from the given list of factors
that are adjacently larger and smaller than n and 
returns that is closest to n  
"""
def get_closest_factor(factors, n):
  factor_list = list(factors)
  factor_list.sort()
  smaller = None
  larger = None
  for f in factor_list:
    if f < n:
      continue
    larger = f 
    break
  for f in reversed(factor_list):
    if f > n: 
      continue
    smaller = f 
    break

  if smaller is None:
    return larger
  elif larger is None:
    return smaller
  else: # return the closest factor
    if abs(smaller - n) <= abs(larger - n):
      return smaller
    else:
      return larger 

    
class PerfStatTracker():
  def __init__(self):
    self.function_time_sums = {}
    self.function_call_counts = {}
    self.function_time_avgs = {}

  def update(self, function, time):
    if not function in self.function_time_sums:
      self.function_time_sums[function] = 0
      self.function_call_counts[function] = 0
      self.function_time_avgs[function] = 0

    self.function_time_sums[function] += time
    self.function_call_counts[function] += 1
    self.function_time_avgs[function] = self.function_time_sums[function]/self.function_call_counts[function]


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  else:
  	print(f"Directory already exists {path}")

  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)

    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
 
def create_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  else:
  	print(f"Directory already exists {path}")

  if scripts_to_save is not None:
    scripts_dir = os.path.join(path, 'scripts')
    if not os.path.exists(scripts_dir):
      os.makedirs(scripts_dir, exist_ok=True)

    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def create_logger(name, level, log_format, log_file):
  handler = logging.FileHandler(log_file)  
  formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')      
  handler.setFormatter(formatter)
  logger = logging.getLogger(name)
  logger.setLevel(level)
  logger.addHandler(handler)
  return logger
 
def get_model_pytorch_file(model_dir, model_version, epoch=0):
  return os.path.join(model_dir, f'model_v{model_version}_epoch{epoch}_pytorch')

def get_model_ast_file(model_dir):
  return os.path.join(model_dir, "model_ast")

# file for storing names of model pytorch and model ast files
def get_model_info_file(model_dir):  
  return os.path.join(model_dir, 'model_info')

def load_model_from_file(model_file, model_version, gpu_id=None):
  with open(model_file, "r") as f:
    ast_file = f.readline().strip()
    pytorch_files = [l.strip() for l in f]

  model_ast = load_ast(ast_file)
  model = model_ast.ast_to_model()
  model.load_state_dict(torch.load(pytorch_files[model_version]))

  if gpu_id:
    model = model.to(device=f"cuda:{gpu_id}")
    model.to_gpu(gpu_id)

  return model, model_ast

def load_ast_from_file(model_file):
  with open(model_file, "r") as f:
    ast_file = f.readline().strip()

  model_ast = load_ast(ast_file)

  return model_ast

def model_id_generator(base_val):
  index = base_val+1
  while True:
    yield index
    index += 1


class ModelManager():
  def __init__(self, model_path, start_id):
    self.base_dir = model_path
    self.start_id = start_id
    self.model_id_generator = model_id_generator(self.start_id)
    
  def load_model(self, model_id, model_version, device=None):
    model_dir = os.path.join(self.base_dir, str(model_id))
    model_info_file = get_model_info_file(model_dir)
    return load_model_from_file(model_info_file, model_version, device)

  def load_model_ast(self, model_id):
    model_dir = os.path.join(self.base_dir, str(model_id))
    model_info_file = get_model_info_file(model_dir)
    model_ast = load_ast_from_file(model_info_file)
    return model_ast 

  def get_next_model_id(self):
    return next(self.model_id_generator)

  def model_dir(self, model_id):
    return os.path.join(self.base_dir, f'{model_id}')

  def save_model_ast(self, model_ast, model_dir):
    ast_file = get_model_ast_file(model_dir)
    model_ast.save_ast(ast_file)

  def save_model_info_file(self, model_dir, model_versions):
    ast_file = get_model_ast_file(model_dir)

    pytorch_files = [get_model_pytorch_file(model_dir, model_version) \
                    for model_version in range(model_versions)]
  
    info_file = get_model_info_file(model_dir)

    with open(info_file, "w+") as f:
      f.write(ast_file + "\n")
      for pf in pytorch_files:
        f.write(pf + "\n")


  def save_model(self, models, model_ast, model_dir):
    if not model_ast is None:
      ast_file = get_model_ast_file(model_dir)
      model_ast.save_ast(ast_file)

    pytorch_files = [get_model_pytorch_file(model_dir, model_version) \
                    for model_version in range(len(models))]
    for i, model in enumerate(models):
      torch.save(model.state_dict(), pytorch_files[i])

    info_file = get_model_info_file(model_dir)

    with open(info_file, "w+") as f:
      if not model_ast is None:
        f.write(ast_file + "\n")
      for pf in pytorch_files:
        f.write(pf + "\n")



