"""
Runs the models produced by search to compute per image psnrs 
or computes the per image psnrs of baseline models whose outputs
are already precomputed 
"""

import argparse
import csv
import os
import torch
import numpy as np
from imageio import imread
import sys
import imageio
import tempfile

rootdir = '/'.join(sys.path[0].split("/")[0:-1])
sys.path.append(rootdir)
pareto_util_dir = os.path.join(rootdir, "pareto-model-selection")
sys.path.append(pareto_util_dir)
baseline_eval_dir = os.path.join(rootdir, "baseline-eval")
sys.path.append(baseline_eval_dir)
sys_run_dir = os.path.join(rootdir, "multinode_sys_run")
sys.path.append(sys_run_dir)

from data_scrape_util import get_model_psnr, get_model_cost
from demosaicnet_models import FullDemosaicknet
from gradienthalide_models import GradientHalide
import demosaic_ast
import torch_model
import demosaicnet
from search_util import insert_green_model

from superres_dataset import SRGBQuadDataset, SBaselineRGBQuadDataset
from superres_only_dataset import SDataset, SBaselineDataset
from xtrans_dataset import XRGBDataset
from dataset import NASDataset, FullPredictionQuadDataset as BayerQuadDataset

import dataset_util 

import logging
logger = logging.getLogger("PIL.PngImagePlugin")
logger.setLevel("ERROR")

import traceback



def get_weight_file(train_dir, training_info_file, search_models=False):
	max_psnr, best_version, best_epoch = get_model_psnr(train_dir, training_info_file, return_epoch=True)
	print(f"model {args.model} using best version {best_version} at epoch {best_epoch}")

	if search_models:
		if best_epoch == 0:
			weight_file = os.path.join(train_dir, f"model_v{best_version}_epoch{best_epoch}_pytorch")
		else:
			weight_file = os.path.join(train_dir, f"model_weights_epoch{best_epoch}")
		print(weight_file)

	else:
		weight_file = os.path.join(train_dir, f"model_v{best_version}_epoch{best_epoch}_pytorch")

	if not os.path.exists(weight_file):
		weight_file = os.path.join(train_dir, f"model_v{best_version}_pytorch")

	return weight_file, max_psnr


def write_dataset_filelist(eval_imagedir, tempfile, is_baseline_dataset):
	with open(tempfile, "w") as tf:
		for f in os.listdir(args.eval_imagedir):
			if is_baseline_dataset:
				if f.endswith("LR.png"):
					tf.write(os.path.join(eval_imagedir, f)+"\n")
			else:
				if f.endswith(".png"):
					tf.write(os.path.join(eval_imagedir, f)+"\n")


parser = argparse.ArgumentParser()
parser.add_argument("--rootdir", type=str, help="root dir for where we store outputs / read inputs")
parser.add_argument("--searchdir", type=str, help="root dir for where search models are stored (use in case retrained models are worse)")
parser.add_argument("--model", type=str, help="name of model")
parser.add_argument("--dataset", type=str, help="name of dataset")
parser.add_argument("--eval_imagedir", type=str, help="the dataset to evaluate on, where the ground truth images are stored")

parser.add_argument("--compute", action='store_true', help="whether we need to recompute model outputs, necessary for our search models")
parser.add_argument('--green_model_asts', type=str, help="file with list of filenames of green model asts")	
parser.add_argument('--green_model_weights', type=str, help="file with list of filenames of green model weights")
parser.add_argument("--crop", type=int)

parser.add_argument("--xtrans_search", action="store_true")
parser.add_argument("--superres_search", action="store_true")
parser.add_argument("--superres_only_search", action="store_true")
parser.add_argument("--bayer_search", action="store_true")
parser.add_argument("--nas_search", action="store_true")

parser.add_argument("--dnetgrid_search", action="store_true")
parser.add_argument("--gradienthalide", action="store_true")
parser.add_argument("--demosaicnet", action="store_true")

args = parser.parse_args()

modeldir = os.path.join(args.rootdir, args.model)
model_outputdir = os.path.join(modeldir, args.dataset)

os.makedirs(model_outputdir, exist_ok=True)
print(model_outputdir)

if args.compute:
	# find the best model weights we have
	if args.demosaicnet or args.gradienthalide:
		train_dir = os.path.join(modeldir, "model_train")
	else:
		train_dir = modeldir

	is_search_model = any([args.xtrans_search, args.superres_search, args.superres_only_search, \
						args.bayer_search, args.nas_search, args.dnetgrid_search])

	training_info_file = os.path.join(train_dir, f"model_{args.model}_training_log")
	weight_file, max_psnr = get_weight_file(train_dir, training_info_file, search_models=is_search_model)

	# see if search produced better psnrs than retraining
	is_retrained = is_search_model and (not args.dnetgrid_search) # dnetgrid directly searched on full dataset, no retraining

	if is_retrained:
		train_dir = os.path.join(args.searchdir, args.model)
		training_info_file = os.path.join(train_dir, f"model_{args.model}_training_log")
		search_weight_file, search_max_psnr = get_weight_file(train_dir, training_info_file, search_models=True)
		if search_max_psnr > max_psnr:
			with open("retrain-v-search.log", "a+") as f:
				f.write(f"{args.model} search psnr {search_max_psnr} retrain psnr {max_psnr}\n")
			weight_file = search_weight_file

	with open("retrain-v-search.log", "a+") as f:
		f.write(f"model {args.model} using weight file {weight_file}\n-----\n")

	if args.demosaicnet:
		params = args.model.split("_")[-1]
		depth = int(params.split("W")[0][1:])
		width = int(params.split("W")[1])
		print(f"eval {args.model} depth {depth} width {width}")
		model = FullDemosaicknet(depth=depth, width=width)
		model.load_state_dict(torch.load(weight_file))
		model.eval()

	elif args.dnetgrid_search:
		depth = int(args.model.split("-")[0])
		width = int(args.model.split("-")[1])
		print(f"eval {args.model} depth {depth} width {width}")
		model = FullDemosaicknet(depth=depth, width=width)
		model.load_state_dict(torch.load(weight_file))
		model.eval()

	elif args.gradienthalide:
		params = args.model.split('-')[-1]
		k = int(params.split("F")[0][1:])
		filters = int(params.split("F")[1])
		print(f"eval {args.model} k {k} filters {filters}")
		model = GradientHalide(k, filters)
		model.load_state_dict(torch.load(weight_file))
		model.eval()
	else:
		ast_file = os.path.join(train_dir, "model_ast")
		model = demosaic_ast.load_ast(ast_file)

		is_factored_search = is_search_model and not (args.superres_only_search or args.nas_search or args.dnetgrid_search)
		if is_factored_search:
			args.green_model_ast_files = [l.strip() for l in open(args.green_model_asts)]
			args.green_model_weight_files = [l.strip() for l in open(args.green_model_weights)]
			insert_green_model(model, args.green_model_ast_files, args.green_model_weight_files)

		print(model.dump())

		model = model.ast_to_model()
		model.load_state_dict(torch.load(weight_file))

	perf_data = {"image":[], "psnr":[]}

	is_baseline_dataset = (args.dataset != "hdrvdp") and (args.dataset != "moire")

	if args.xtrans_search:
		DatasetType = XRGBDataset
	elif args.bayer_search:
		DatasetType = BayerQuadDataset
	elif args.superres_search:
		if is_baseline_dataset:
			print(f"USING BASELINE DATASET")
			DatasetType = SBaselineRGBQuadDataset
		else:
			DatasetType = SRGBQuadDataset
	elif args.superres_only_search:
		if is_baseline_dataset:
			print(f"USING BASELINE DATASET")
			DatasetType = SBaselineDataset
		else:
			DatasetType = SDataset
	else: # nas, dnetgrid, gradhalide, demosaicnet
		DatasetType = NASDataset

	dataset_filelist_f = "_temp.txt"
	write_dataset_filelist(args.eval_imagedir, dataset_filelist_f, not (args.dataset == "hdrvdp" or args.dataset == "moire"))
	image_files = dataset_util.ids_from_file(dataset_filelist_f)

	dataset = DatasetType(dataset_filelist_f, return_index=True)
	indices = list(range(len(dataset)))
	sampler = torch.utils.data.sampler.SequentialSampler(indices)

	data_queue = dataset_util.FastDataLoader(
					dataset, batch_size=1,
					sampler=sampler,
					pin_memory=True, num_workers=1)

	with torch.no_grad():
		for index, input, target in data_queue:
			image_f = image_files[index]
			subdir, image_id = dataset_util.get_image_id(image_f)

			if args.xtrans_search:
				(packed_mosaic, mosaic3chan, flat_mosaic, rb) = input
				model_inputs = {"Input(Mosaic)": mosaic3chan,
					"Input(Mosaic3x3)": packed_mosaic, 
					"Input(FlatMosaic)": flat_mosaic,
					"Input(RBXtrans)": rb}     
			elif args.superres_search:
				(bayer_quad, redblue_bayer) = input
				img = target
				model_inputs = {"Input(Mosaic)": bayer_quad, 
					"Input(RedBlueBayer)": redblue_bayer}
			elif args.superres_only_search:
				model_inputs = {"Input(Image)": input}
			elif args.bayer_search:
				bayer_quad, redblue_bayer, green_grgb = input
				model_inputs = {"Input(Mosaic)": bayer_quad, 
					"Input(Green@GrGb)": green_grgb, 
					"Input(RedBlueBayer)": redblue_bayer}
			elif args.nas_search:
				model_inputs = {"Input(Mosaic)": input}
			elif args.gradienthalide:
				bayer = input
				flatbayer = torch.sum(bayer, axis=1, keepdims=True)
				input = (flatbayer, bayer)
			elif args.demosaicnet or args.dnetgrid_models:
				pass # no futher data prep needed, just takes the 3 channel bayer input
			else:
				assert False, "Unspecified model type"

			is_ast_pytorch_model = is_search_model and (not args.dnetgrid_search)
			try:
				if is_ast_pytorch_model:
					model.reset()
					output = model.run(model_inputs)
					# print(f"ran model...")
					# print(f"model output shape {output.shape}")
					# print(f"target shape {target.shape}")
				else:
					output = model(input)

				clamped = torch.clamp(output, 0, 1)

				# can't afford to save all search model outputs, 100 models per search 
				if not is_search_model:
					out_img = dataset_util.tensor2image(clamped)
					outfile = os.path.join(model_outputdir, image_f)
					print(outfile)
					imageio.imsave(outfile, out_img)

				clamped = clamped.squeeze(0)
				target = target.squeeze(0)
				if args.crop > 0:
					crop = args.crop
					clamped = clamped[:,crop:-crop,crop:-crop]
					target = target[:,crop:-crop,crop:-crop]

				# print(f"after crop img {img.shape} out {clamped.shape}")
				# print(f"output shape {output.shape}")

				image_mse = (clamped-target).square().mean(-1).mean(-1).mean(-1)
				image_psnr = -10.0*torch.log10(image_mse)

				perf_data["image"].append(image_f)
				perf_data["psnr"].append(image_psnr.item())
		     
				print(f"psnr: {image_psnr}")

			except Exception as e:
				with open("failed_sr_search_models.txt", "a+") as f:
					f.write(f"{args.model} {args.dataset}")
					f.write(f"{e}\n")

	os.remove(dataset_filelist_f)

else: # evaluate precomputed image psnrs
	perf_data = {"image":[], "psnr":[]}

	for f in os.listdir(model_outputdir):
		image_id = f.split("-")[-1]

		ground_truth_f = os.path.join(args.eval_imagedir, image_id)
		output_f = os.path.join(model_outputdir, f)

		if not os.path.exists(ground_truth_f): # henz outputs messed up names
			if args.dataset == "mcm":
				image_id = f.split(".")[0]
				ground_truth_f = os.path.join(args.eval_imagedir, f"mcm_{image_id}.png")
			elif args.dataset == "kodak":
				image_id = f.split('.')[0][-2:]
				image_id = image_id.zfill(4)
				ground_truth_f = os.path.join(args.eval_imagedir, f"IMG{image_id}.png")
		print(ground_truth_f)
		print(output_f)
		
		img = torch.tensor(np.array(imread(ground_truth_f)).astype(np.float32) / (2**8-1))
		output_img = torch.tensor(np.array(imread(output_f)).astype(np.float32) / (2**8-1))

		if args.xtrans:
			img = img[4:-4,4:-4,:]

		if args.crop > 0:
			crop = args.crop
			img = img[crop:-crop,crop:-crop,:]
			output_img = output_img[crop:-crop,crop:-crop,:]

		clamped = torch.clamp(output_img, min=0, max=1).detach()
		image_mse = (clamped-img).square().mean(-1).mean(-1).mean(-1)
		image_psnr = -10.0*torch.log10(image_mse)

		perf_data["image"].append(image_id)
		perf_data["psnr"].append(image_psnr.item())


avg_psnr = np.mean(perf_data["psnr"])

psnr_filename = os.path.join(modeldir, f"{args.dataset}_psnrs.csv")
print(psnr_filename)
avg_psnr_filename = os.path.join(modeldir, f"{args.dataset}_avg_psnr.txt")

with open(psnr_filename, "w") as f:
	writer = csv.DictWriter(f, fieldnames=["image", "psnr"])
	writer.writeheader()
	for i in range(len(perf_data["image"])):
		writer.writerow({"image": perf_data["image"][i], "psnr": perf_data["psnr"][i]})
with open(avg_psnr_filename, "w") as f:
	f.write(f"{avg_psnr}\n")


