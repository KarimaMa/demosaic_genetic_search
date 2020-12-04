import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import csv
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
	parser.add_argument("--csv_files", type=str, help='model database csv')
	parser.add_argument("--model_dir", type=str, help="directory with model info")
	parser.add_argument("--pareto_ids", type=str, help="file with list of pareto model ids")
	parser.add_argument("--visfolder", type=str, help="where to save graph drawings")
	parser.add_argument("--model_inits", type=int )
	parser.add_argument("--table_w", type=int)
	parser.add_argument("--table_h", type=int)
	parser.add_argument("--num_tables", type=int)
	args = parser.parse_args()

	cost_evaluator = ModelEvaluator(None)
	
	csv_files = [f.strip() for f in args.csv_files.split(",")]
	modeldb_info = {}
	for i,f in enumerate(csv_files):
		with open(f, newline='\n') as csvf:
			reader = csv.DictReader(csvf, delimiter=',')
			for r in reader:	
				psnrs = [float(p) for p in [r["psnr_0"], r["psnr_1"], r["psnr_2"]]]
				best_psnr = max(psnrs)
				modeldb_info[r["model_id"]] = best_psnr
				
	model_info = {}
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

			graph = vis_ast(model, f"pareto_model_{model_id}")
			visname = f"{args.visfolder}/pareto_model_{model_id}"
			vispng = visname + ".png"
			graph.render(visname)
			model_info[model_id] = {"cost":cost , "psnr":best_psnr, "psnr6e":modeldb_info[model_id], "image":vispng}

	n_cells = args.table_w * args.table_h
	for i, model_id in enumerate(model_info):
		if i % (n_cells) == 0:
			print(f"{i} creating new table")
			# create new table 
			plt.figure(figsize=(args.table_h,args.table_w))
			gs = gridspec.GridSpec(args.table_h, args.table_w)
			gs.update(wspace=0.025, hspace=0.01) # set the spacing between axes. 
			for c in range(n_cells):
				ax = plt.subplot(gs[c])
				ax.axis('off')

		cost = model_info[model_id]["cost"]
		psnr = model_info[model_id]["psnr"]
		psnr6e = model_info[model_id]["psnr6e"]
		imgfile = model_info[model_id]["image"]		

		ax = plt.subplot(gs[i % n_cells])
		ax.imshow(plt.imread(imgfile))
		textstr = f"model {model_id}\ncost {cost}\npsnr {psnr:.2f}\npsnr6e {psnr6e:.2f}"
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		ax.text(0.8, 0.05, textstr, transform=ax.transAxes, fontsize=6, verticalalignment='top', bbox=props)

		if i % (n_cells) == n_cells-1  or i == len(model_info) - 1:
			plt.show()
			plt.savefig(os.path.join(args.visfolder, "table_{i}"))
			plt.clf()

