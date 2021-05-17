import os
import csv
import argparse

import sys

sys.path.append(sys.path[0].split("/")[0])

import cost
import demosaic_ast

"""
figures out average PSNR lost per model if we only keep 
training the best initialization after epoch 1
"""

"""
returns best PSNR seen up to the given epoch
"""
def get_best_psnr(file, cutoff_epoch=None):
  with open(file, "r") as f:
    if cutoff_epoch is None:
      psnrs = [float(l.strip().split(' ')[-1]) for l in f]
    else:
      psnrs = [float(l.strip().split(' ')[-1]) for i,l in enumerate(f) if i < cutoff_epoch]
    if len(psnrs) == 0:
      return None
    return max(psnrs)

"""
returns the best init based on PSNRs see up to the 
given epoch
"""
def get_best_init(modeldir, modelid, inits, cutoff_epoch):
  best_init = -1
  best_psnr = -1
  for i in range(inits):
    file = os.path.join(modeldir, f"v{i}_validation_log")
    if not os.path.exists(file):
      return None

    best_psnr_i = get_best_psnr(file, cutoff_epoch)

    if best_psnr_i is None:
      return None
    if best_psnr_i > best_psnr:
      best_init = i
      best_psnr = best_psnr_i
  return best_init


def get_best_psnr_across_inits(modeldir, inits):
  psnrs = []
  for i in range(inits):
    file = os.path.join(modeldir, f"v{i}_validation_log")
    if not os.path.exists(file):
      return None

    best_psnr_i = get_best_psnr(file)
    if not best_psnr_i is None:
      psnrs.append(best_psnr_i)
  if len(psnrs) == inits:
    return max(psnrs)
  else:
    return None


"""
returns PSNR delta between the best PSNR of the init chosen
based on the best looking init by cutoff epoch and the best
PSNR across all inits
"""
def get_lost_psnr(modeldir, modelid, inits, cutoff_epoch):
  best_possible_psnr = get_best_psnr_across_inits(modeldir, inits)
  best_init = get_best_init(modeldir, modelid,  inits, cutoff_epoch)
  if best_init is None:
    return None, None

  init_file = os.path.join(modeldir, f"v{best_init}_validation_log")
  chosen_init_best_psnr = get_best_psnr(init_file)
  lost_psnr = best_possible_psnr - chosen_init_best_psnr
  return lost_psnr, best_possible_psnr


def write_csv(model_rootdir, inits, outfile, cutoff_epoch):
  with open(outfile, "w") as f:
    fieldnames = ['modelid', 'lost_psnr', 'best_possible_psnr', 'cost']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    costmodel = cost.ModelEvaluator(None)

    for modelid in os.listdir(model_rootdir):
      modeldir = os.path.join(model_rootdir, modelid)
      ast_file = os.path.join(modeldir, "model_ast")
      ast = demosaic_ast.load_ast(ast_file)
      modelcost = costmodel.compute_cost(ast)

      lost_psnr, best_possible_psnr = get_lost_psnr(modeldir, modelid, inits, cutoff_epoch)
      if lost_psnr is None:
        continue
      writer.writerow({'modelid': modelid, 'lost_psnr': lost_psnr, 'best_possible_psnr': best_possible_psnr, 'cost': modelcost})


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_root", type=str)
  parser.add_argument("--inits", type=int, default=3)
  parser.add_argument("--outfile", type=str)
  parser.add_argument("--cutoff_epoch", type=int, default=1)
  args = parser.parse_args()

  write_csv(args.model_root, args.inits, args.outfile, args.cutoff_epoch)



