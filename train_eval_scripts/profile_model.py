import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.utils
import torch.autograd.profiler as profiler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os
import numpy as np
import sys
import re
import csv
sys.path.append(sys.path[0].split("/")[0])

from dataset import GreenQuadDataset, FullPredictionQuadDataset, FastDataLoader, FullPredictionProcessedDataset, GreenProcessedQuadDataset
import util
from demosaic_ast import load_ast, get_green_model_id, Input
import cost


def run(args, model, model_id):
  if args.gpu >= 0: 
    model = model.cuda()
    model.to_gpu(args.gpu)
    
  if not args.full_model:
    test_data = GreenProcessedQuadDataset(data_file=args.test_file, return_index=True)
  else:
    test_data = FullPredictionProcessedDataset(data_file=args.test_file, RAM=True, return_index=True)

  infer(args, test_data, model, model_id)


def infer(args, test_data, model, model_id):
  num_test = len(test_data)
  test_indices = list(range(num_test))

  test_queue = FastDataLoader(
    test_data, batch_size=args.batchsize,
    sampler=torch.utils.data.sampler.SequentialSampler(test_indices),
    pin_memory=False, num_workers=4)
  
  criterion = nn.MSELoss()

  model.eval()

  with torch.no_grad():
    with profiler.profile(use_cuda=True) as prof:
      with profiler.record_function("model_inference"):
        for step, (input_img_names, input, target) in enumerate(test_queue):
          if args.full_model:
            bayer, redblue_bayer, green_grgb = input 
            redblue_bayer = redblue_bayer.float()
            green_grgb = green_grgb.float()
            if args.gpu >= 0:
              redblue_bayer = redblue_bayer.cuda()
              green_grgb = green_grgb.cuda()
          else:
            bayer = input
            
          bayer = bayer.float()
          target = target.float()
          if args.gpu >= 0:
            bayer = bayer.cuda()
            target = target.cuda()
          
          target = target[..., args.crop:-args.crop, args.crop:-args.crop]

          n = bayer.size(0)
          model.reset()
          if args.full_model:
            model_inputs = {"Input(Bayer)": bayer, 
                            "Input(Green@GrGb)": green_grgb, 
                            "Input(RedBlueBayer)": redblue_bayer}
            model.run(model_inputs)
          else:
            model_inputs = {"Input(Bayer)": bayer}
            model.run(model_inputs)

  prof.export_chrome_trace(f"model-{model_id}-trace.json")
  print(prof.key_averages().table(row_limit=25))


"""
Inserts green ast into chroma ast and sets 
no_grad for all sub models inside the chroma ast
"""
def insert_green_model(args, chroma_ast, green_ast):
  nodes = chroma_ast.preorder()
  for n in nodes:
    if type(n) is Input:
      if n.name == "Input(GreenExtractor)":
        n.node = green_ast
        n.no_grad = True
      elif hasattr(n, "node"): # other input ops my run submodels that also require the green model
        insert_green_model(args, n.node, green_ast)
        n.no_grad = True


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Demosaic")
  parser.add_argument('--modeldir', type=str, help='where model asts, training logs, and weights are stored')
  parser.add_argument('--model_id', type=int, help='model id from modeldir to run')
  parser.add_argument('--crop', type=int, default=16)
  parser.add_argument('--batchsize', type=int, default=64)
  parser.add_argument('--gpu', type=int, default=-1)
  
  # training parameters
  parser.add_argument('--test_file', type=str, help='filename of file with list of data image files to evaluate')
  
  # training full chroma + green parameters
  parser.add_argument('--full_model', action="store_true")
  parser.add_argument('--modular', action="store_true")

  parser.add_argument('--green_model_asts', type=str, help="file with list of filenames of green model asts used as inputs to chroma models")

  args = parser.parse_args()

  if args.gpu >= 0 and not torch.cuda.is_available():
    sys.exit(1)
  
  torch.cuda.set_device(args.gpu)

  if args.full_model and args.modular:
    args.green_model_asts = [l.strip() for l in open(args.green_model_asts)]

  model_id = args.model_id
  print(f"running model {model_id}")
  model_info_dir = os.path.join(args.modeldir, f"{model_id}")
  model_ast_file = os.path.join(model_info_dir, "model_ast")

  model_ast = load_ast(model_ast_file)

  if args.full_model and args.modular:
    green_model_id = get_green_model_id(model_ast)
    green_model_ast_file = args.green_model_asts[green_model_id]
    green_model = load_ast(green_model_ast_file)
    
    insert_green_model(args, model_ast, green_model)

    print(model_ast.dump())

  if args.gpu >= 0:  
    torch_model = model_ast.ast_to_model().cuda() 
  else:
    torch_model = model_ast.ast_to_model()

  torch_model._initialize_parameters()
  
  run(args, torch_model, model_id)
