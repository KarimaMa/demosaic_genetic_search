import csv
import sys
import logging
from database import Database
import os
import argparse
from type_check import count_parameterized_depth
from demosaic_ast import load_ast
sys.path.append(os.path.join(sys.path[0], "sys_run"))
from cost import CostTiers
import util

def parse_cost_tiers(s):
  ranges = s.split(' ')
  ranges = [[int(x) for x in r.split(',')] for r in ranges]
  print(ranges)
  return ranges

parser = argparse.ArgumentParser("Demosaic")
parser.add_argument('--model_db_csv', type=str)
parser.add_argument('--model_parentdir', type=str)
parser.add_argument('--max_gen', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--model_inits', type=int)

args = parser.parse_args()

args.cost_tiers = parse_cost_tiers(args.cost_tiers)

log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
  format=log_format, datefmt='%m/%d %I:%M:%S %p')

logger = util.create_logger('logger', logging.INFO, log_format, 'foo_log')

fields = ["model_id", "generation", "tier", "depth", "epoch0", "epoch1", "epoch2"]
field_types = [int, int, int, int, float, float, float]
util.create_dir("val_psnr_database")
val_psnr_db = Database('ValPSNR_Databse', fields, field_types, "val_psnr_database")
val_psnr_db.cntr = 0

def build_model_database(model_inits):
  fields = ["model_id", "id_str", "hash", "structural_hash", "generation", "occurrences", "best_init"]
  field_types = [int, str, int, int, int, int, int]
  for model_init in range(model_inits):
    fields += [f"psnr_{model_init}"]
    field_types += [float]
  fields += ["compute_cost", "parent_id", "failed_births", "failed_mutations",\
             "prune_rejections", "structural_rejections", "seen_rejections"]
  field_types += [float, int, int, int, int, int, int]
  return Database("ModelDatabase", fields, field_types, args.model_database_dir)

model_db = build_model_database(3)
model_db.load(args.model_db_csv)

for model_id in model_db.table:
  model_dir = os.path.join(args.model_parentdir, f"{model_id}")
  init_psnrs = []
  data = {}
  ast_file = os.path.join(model_dir, "model_ast")
  ast = load_ast(ast_file)
  depth = count_parameterized_depth(ast)

  for init in range(args.model_inits):
    if model_id == 0:
      psnrs = [31.38 for i in range(args.epochs)]
    else:
      psnr_file = os.path.join(model_dir, f"v{init}_validation_log")
      if not os.path.exists(psnr_file):
        continue
      else:
        psnrs = []
        for l in open(psnr_file, "r"):
          loss = float(l.split(' ')[-1].strip())
          psnr = util.compute_psnr(loss)
          psnrs.append(psnr)
        psnrs = psnrs[-3:]
    print(f"model {model_id} init {init} {psnrs}")
    if len(psnrs) == 0:
      continue
    init_psnrs.append(psnrs)

  if len(init_psnrs) == 0:
    continue
  final_psnrs = [psnrs[-1] for psnrs in init_psnrs]
  best_psnr = max(final_psnrs)
  best_init = final_psnrs.index(best_psnr)

  used_psnrs = init_psnrs[best_init]
  data["model_id"] = model_id
  data["depth"] = depth

  for init in range(args.model_inits):
    data[f"epoch{init}"] = used_psnrs[init]

  val_psnr_db.add(val_psnr_db.cntr, data)
  val_psnr_db.cntr += 1

val_psnr_db.save()



