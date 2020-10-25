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

def parse_cost_tiers(s):
  ranges = s.split(' ')
  ranges = [[int(x) for x in r.split(',')] for r in ranges]
  print(ranges)
  return ranges


parser = argparse.ArgumentParser("Demosaic")
parser.add_argument('--tier_database_file', type=str, default='cost_tier_database')
parser.add_argument('--cost_tiers', type=str, help='list of tuples of cost tier ranges')
parser.add_argument('--model_parentdir', type=str)
parser.add_argument('--max_gen', type=int)
parser.add_argument('--output_dir', type=str, help='where to store model info')
parser.add_argument('--model_depth_list_dir', type=str, help='where to store model list with depth info')

args = parser.parse_args()

if not os.path.exists(args.model_depth_list_dir):
	util.create_dir(args.model_depth_list_dir)

args.cost_tiers = parse_cost_tiers(args.cost_tiers)

log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
  format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = util.create_logger('logger', logging.INFO, log_format, 'foo_log')

models_by_depth = {}

for gen in range(args.max_gen+1):
	cost_tiers = CostTiers('junk', args.cost_tiers, logger)
	cost_tiers.load_generation_from_database(args.tier_database_file, gen)

	for tid, tier in enumerate(cost_tiers.tiers):
		for model_id in tier:
			model_dir = os.path.join(args.model_parentdir, f"{model_id}")
			init_psnrs = []
			data = {}
			ast_file = os.path.join(model_dir, "model_ast")
			ast = load_ast(ast_file)
			depth = count_parameterized_depth(ast)

			data["model_id"] = model_id
			data["generation"] = gen
			data["tier"] = tid
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



