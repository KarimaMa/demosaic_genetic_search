import argparse
import shutil
import os


parser = argparse.ArgumentParser()
parser.add_argument("--model_ids", type=str, help="list of pareto models")
parser.add_argument("--outdir", type=str, help="where to copy over pareto models to retrain")
parser.add_argument("--indir", type=str, help="where search models are")

args = parser.parse_args()

model_ids = [l.strip() for l in open(args.model_ids)]

os.makedirs(args.outdir, exist_ok=True)

for model_id in model_ids:
	shutil.copytree(os.path.join(args.indir, model_id), os.path.join(args.outdir, model_id))