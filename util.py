import logging
import torch
import os
from demosaic_ast import load_ast

# from a github gist by victorlei
def extclass(cls):
  return lambda f: (setattr(cls,f.__name__,f) or f)

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
  if os.path.exists(path):
  	assert False, f"Attempting to overwrite existing experiment {path}"
  
  os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
 
def create_dir(path):
  if os.path.exists(path):
    assert False, f"Attempting to overwrite existing folder {path}"

  os.makedirs(path)
  print('Created folder: {}'.format(path))


def create_logger(name, level, log_format, log_file):
  handler = logging.FileHandler(log_file)  
  formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')      
  handler.setFormatter(formatter)
  logger = logging.getLogger(name)
  logger.setLevel(level)
  logger.addHandler(handler)
  return logger
 
def get_model_pytorch_file(model_dir, model_version):
  return os.path.join(model_dir, f'model_v{model_version}_pytorch')

def get_model_ast_file(model_dir):
  return os.path.join(model_dir, "model_ast")

# file for storing names of model pytorch and model ast files
def get_model_info_file(model_dir):  
  return os.path.join(model_dir, 'model_info')

def load_model_from_file(model_file, model_version):
  with open(model_file, "r") as f:
    ast_file = f.readline().strip()
    pytorch_files = [l.strip() for l in f]

  model = torch.load(pytorch_files[model_version])
  model_ast = load_ast(ast_file)
  return model, model_ast


def model_id_generator():
  index = 1
  while True:
    yield index
    index += 1


class ModelManager():
  def __init__(self, model_path):
    self.base_dir = model_path
    self.model_id_generator = model_id_generator()

  def load_model(self, model_name, model_version):
    model_dir = os.path.join(self.base_dir, model_name)
    model_info_file = get_model_info_file(model_dir)
    return load_model_from_file(model_info_file, model_version)

  def get_next_model_id(self):
    next(self.model_id_generator)

  def model_dir(self, model_id):
    return os.path.join(self.base_dir, f'{model_id}')

  def save_model(self, models, model_ast, model_dir):
    ast_file = get_model_ast_file(model_dir)
    model_ast.save_ast(ast_file)

    pytorch_files = [get_model_pytorch_file(model_dir, model_version) \
                    for model_version in range(len(models))]
    for i, model in enumerate(models):
      torch.save(model, pytorch_files[i])

    info_file = get_model_info_file(model_dir)
    with open(info_file, "w+") as f:
      f.write(ast_file + "\n")
      for pf in pytorch_files:
        f.write(pf)



