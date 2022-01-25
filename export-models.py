import demosaic_ast
import argparse
import sys
import os
import torch
import torch_model
import torch.nn as nn
import simplified_torch_model


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
parser.add_argument("--modelid", type=str, help="model id to save")
parser.add_argument("--retrain_dir", type=str, help="retrained models dir")
parser.add_argument("--search_dir", type=str, help="search models dir")

args = parser.parse_args()

args.green_model_ast_files = [l.strip() for l in open(args.green_model_asts)]
args.green_model_weight_files = [l.strip() for l in open(args.green_model_weights)]

m = args.modelid

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
print(model.dump())

model = model.ast_to_model()
model.load_state_dict(torch.load(weight_file))
os.makedirs(args.outputdir, exist_ok=True)
torch.save(model, f"{args.outputdir}/{m}.pt")

img_w = 120
Mosaic = torch.rand((1,3,img_w,img_w), requires_grad=False).float()
Mosaic3x3 = torch.rand((1,36,img_w//6,img_w//6), requires_grad=False).float()
FlatMosaic = torch.rand((1,1,img_w,img_w), requires_grad=False).float()
RBXtrans = torch.rand((1,16,img_w//6,img_w//6), requires_grad=False).float()

sample_input = {"Input(Mosaic)": Mosaic,
				"Input(Mosaic3x3)": Mosaic3x3,
                "Input(FlatMosaic)": FlatMosaic,
                "Input(RBXtrans)": RBXtrans}
model.eval()
ids2inputs = {}
calling_order = []
model.get_calling_order(sample_input, ids2inputs, calling_order)

seen_ids = set()
member_field_str = ""
class_header_str = \
	"class TorchModel(nn.Module):\n\tdef __init__(self):\n\t\tsuper(TorchModel, self).__init__()\n"

for node_id, node_obj in calling_order:
	if not isinstance(node_obj, torch_model.InputOp) and not node_id in seen_ids:
		member_field_str += f"\t\t{node_obj.genfieldname(node_id)}\n"
		seen_ids.add(node_id)

seen_ids = set()
inputs2varnames = {}
forward_str = f"\tdef forward(self, Mosaic, Mosaic3x3, FlatMosaic, RBXtrans):\n"
for node_id, node_obj in calling_order:
	if not node_id in seen_ids:
		forward_str += f"\t\t{node_obj.genrun(node_id, ids2inputs, inputs2varnames)}\n"
		seen_ids.add(node_id)
last_output = calling_order[-1][0] + "_out"
forward_str += f"\t\treturn {last_output}\n"

class_str = class_header_str + member_field_str + forward_str
exec(class_str)
print(class_str)

simple_model = TorchModel()
node2param = {}
for node_id, node_obj in calling_order:
	node_obj.map_node2param(node_id, node2param)
for node_id, node_obj in calling_order:
	simplified_torch_model.set_parameters(simple_model, node_id, node2param)

traced_model = torch.jit.trace(simple_model, example_inputs=(Mosaic, Mosaic3x3, FlatMosaic, RBXtrans))
print("TRACED!")

correct_output = model.run(sample_input)

print("running traced model...")
traced_output = traced_model(Mosaic, Mosaic3x3, FlatMosaic, RBXtrans)
print(f"diff between traced model output and correct model output {(traced_output-correct_output).pow(2).sum()}")

print(f"saving traced model...")
torch.jit.save(traced_model, os.path.join(args.outputdir, f'{m}.pt'))




