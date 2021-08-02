import demosaic_ast
import argparse
import sys
import os
import torch
import torch_model

sys_run_dir = "./multinode_sys_run"
sys.path.append(sys_run_dir)
pareto_util_dir = "pareto-model-selection"
sys.path.append(pareto_util_dir)

from search_util import insert_green_model
from data_scrape_util import get_model_psnr


def get_weight_file(train_dir, training_info_file, search_models=False):
	max_psnr, best_version, best_epoch = get_model_psnr(train_dir, training_info_file, return_epoch=True)
	print(f"model from {train_dir} using best version {best_version} at epoch {best_epoch}")

	if search_models:
		if best_epoch == 0:
			weight_file = os.path.join(train_dir, f"model_v{best_version}_epoch{best_epoch}_pytorch")
		else:
			weight_file = os.path.join(train_dir, f"model_weights_epoch{best_epoch}")

	else:
		weight_file = os.path.join(train_dir, f"model_v{best_version}_epoch{best_epoch}_pytorch")

	if not os.path.exists(weight_file):
		weight_file = os.path.join(train_dir, f"model_v{best_version}_pytorch")

	return weight_file, max_psnr



parser = argparse.ArgumentParser()
parser.add_argument('--green_model_asts', type=str, help="file with list of filenames of green model asts")	
parser.add_argument('--green_model_weights', type=str, help="file with list of filenames of green model weights")
parser.add_argument('--outputdir', type=str)
parser.add_argument("--models", type=str, help="model ids to save")
parser.add_argument("--retrain_dir", type=str, help="retrained models dir")
parser.add_argument("--search_dir", type=str, help="search models dir")

args = parser.parse_args()

args.green_model_ast_files = [l.strip() for l in open(args.green_model_asts)]
args.green_model_weight_files = [l.strip() for l in open(args.green_model_weights)]
args.models = args.models.split(",")

for m in args.models:
	retrain_dir = os.path.join(args.retrain_dir, m)
	retrain_info_file = os.path.join(retrain_dir, f"model_{m}_training_log")
	retrain_weight_file, retrain_max_psnr = get_weight_file(retrain_dir, retrain_info_file, search_models=True)

	search_dir = os.path.join(args.search_dir, m)
	search_info_file = os.path.join(search_dir, f"model_{m}_training_log")
	search_weight_file, search_max_psnr = get_weight_file(search_dir, search_info_file, search_models=True)

	if search_max_psnr > retrain_max_psnr:
		weight_file = search_weight_file
	else:
		weight_file = retrain_weight_file

	print(f"{m} search psnr {search_max_psnr} retrain psnr {retrain_max_psnr} weight {weight_file}\n")

	ast_file = os.path.join(retrain_dir, "model_ast")
	model = demosaic_ast.load_ast(ast_file)

	insert_green_model(model, args.green_model_ast_files, args.green_model_weight_files)
	model = model.ast_to_model()
	model.load_state_dict(torch.load(weight_file))
	torch.save(model, f"{args.outputdir}/{m}.pt")

	img_w = 60
	sample_input = {"Input(Mosaic)": torch.zeros((1,3,img_w,img_w), requires_grad=False).float(),
					"Input(Mosaic3x3)": torch.zeros((1,36,img_w//6,img_w//6), requires_grad=False).float(),
                    "Input(FlatMosaic)": torch.zeros((1,1,img_w,img_w), requires_grad=False).float(),
                    "Input(RBXtrans)": torch.zeros((1,16,img_w//6,img_w//6), requires_grad=False).float()}
	model.eval()
	with torch.no_grad():            
		traced_gpu = torch.jit.trace(model, sample_input)
		torch.jit.save(traced_gpu, f"{args.outputdir}/{m}_gpu.pt")

		cpu_model = model.cpu()
		sample_input_cpu = sample_input.cpu()
		traced_cpu = torch.jit.trace(cpu_model, sample_input_cpu)
		torch.jit.save(traced_cpu, f"{args.outputdir}/{m}_cpu.pt")





