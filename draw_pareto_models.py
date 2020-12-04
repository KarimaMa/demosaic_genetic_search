import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math
import os
from graph_viz import *
import argparse
import demosaic_ast
from cost import ModelEvaluator


def compute_psnr(loss):
  return 10*math.log(math.pow(255,2) / math.pow(math.sqrt(loss)*255, 2),10)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_dir", type=str, help="directory with model info")
	parser.add_argument("--pareto_ids", type=str, help="file with list of pareto model ids")
	parser.add_argument("--visfolder", type=str, help="where to save graph drawings")
	parser.add_argument("--model_inits", type=int )
	parser.add_argument("--table_w", type=int)
	parser.add_argument("--table_h", type=int)
	args = parser.parse_args()

	f, axarr = plt.subplots(args.table_h, args.table_w)

	cost_evaluator = ModelEvaluator(None)

	with open(args.pareto_ids, "r") as f:
		for i,l in enumerate(f):
			model_id = l.strip()
			model_subdir = os.path.join(args.model_dir, f"{model_id}")
			model = demosaic_ast.load_ast(f"{model_subdir}/model_ast")

			cost = cost_evaluator.compute_cost(model)
			
			init_psnrs = []
			for init in range(args.model_inits):
				loss_file = os.path.join(model_subdir, f"v{init}_validation_log")
				if not os.path.exists(loss_file):
					print(f"model id {model_id} does not have validation v{init} info")
					continue
				else:
					psnrs = []
					for l in open(loss_file, "r"):
						loss = float(l.split(' ')[-1].strip())
						retrain_psnr = compute_psnr(loss)
						psnrs.append(retrain_psnr)

					if len(psnrs) == 0:
						continue
					best_psnr = max(psnrs)

				init_psnrs.append(best_psnr)

			if len(init_psnrs) == 0:
				continue

			best_psnr = max(init_psnrs)
			print(f"model {model_id} cost {cost} best psnr {best_psnr}")

			graph = vis_ast(model, f"pareto_model_{model_id}")
			visname = f"{args.visfolder}/pareto_model_{model_id}"
			vispng = visname + ".png"
			graph.render(visname)

			xloc = i % args.table_w
			yloc = i // args.table_h 
			axarr[yloc, xloc].imshow(plt.imread(vispng))
	

	plt.savefig(os.path.join(args.visfolder, "table"))

