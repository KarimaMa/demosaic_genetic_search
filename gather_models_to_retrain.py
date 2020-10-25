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
import shutil


parser = argparse.ArgumentParser("Demosaic")
parser.add_argument('--model_db_csv', type=str, default='model database')
parser.add_argument('--model_parentdir', type=str)
parser.add_argument('--max_gen', type=int)
parser.add_argument('--output_dir', type=str, help='where to store model info')
parser.add_argument('--model_depth_list_dir', type=str, help='where to store model list with depth info')

args = parser.parse_args()

if not os.path.exists(args.model_depth_list_dir):
	util.create_dir(args.model_depth_list_dir)

log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
  format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = util.create_logger('logger', logging.INFO, log_format, 'foo_log')

models_by_depth = {}

def build_model_database(args):
  fields = ["model_id", "id_str", "hash", "structural_hash", "generation", "occurrences", "best_init"]
  field_types = [int, str, int, int, int, int, int]
  for model_init in range(args.model_inits):
    fields += [f"psnr_{model_init}"]
    field_types += [float]
  fields += ["compute_cost", "parent_id", "failed_births", "failed_mutations",\
             "prune_rejections", "structural_rejections", "seen_rejections"]
  field_types += [float, int, int, int, int, int, int]
  return Database("ModelDatabase", fields, field_types, 'junk')

model_db = build_model_database(args)
model_db.load(args.model_db_csv)

for model_id in model_db.table:
	model_dir = os.path.join(args.model_parentdir, f"{model_id}")
	init_psnrs = []
	data = {}
	ast_file = os.path.join(model_dir, "model_ast")
	ast = load_ast(ast_file)
	depth = count_parameterized_depth(ast)

	data["model_id"] = model_id
	data["depth"] = depth

	if depth in models_by_depth:
		models_by_depth[depth].append(data)
	else:
		models_by_depth[depth] = [data]
	depth_filename = f"{args.model_depth_list_dir}/depth_{depth}_models.txt"

	with open(depth_filename, "a+") as f:
		f.write(f"{model_id}\n")

# sample each depth for models to retrain
max_depth = max(list(models_by_depth.keys()))
min_depth = min(list(models_by_depth.keys()))

util.create_dir(args.output_dir)

for d in range(min_depth, max_depth):
	if d in models_by_depth:
		for mdata in models_by_depth[d]:
			model_id = mdata["model_id"]
			model_dir = os.path.join(args.model_parentdir, f"{model_id}")
			ast_file = os.path.join(model_dir, "model_ast")
			new_model_dir = os.path.join(args.output_dir, f"{model_id}")
			util.create_dir(new_model_dir)
			shutil.copyfile(ast_file, os.path.join(new_model_dir, "model_ast"))



