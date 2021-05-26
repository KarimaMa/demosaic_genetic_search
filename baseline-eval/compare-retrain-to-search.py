import argparse
import csv
import os
import torch
import numpy as np
from imageio import imread
import sys
import imageio

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
import demosaicnet
from run_with_dataset import tensor2image
from search_util import insert_green_model

"""
For demosaicking models
"""

# takes 3D batch input 
def gen_model_inputs(mosaic):
	mosaic = torch.sum(mosaic, axis=0, keepdims=True)
	image_size = list(mosaic.shape)
	quad_h = image_size[1] // 2
	quad_w = image_size[2] // 2

	redblue_bayer = np.zeros((2, quad_h, quad_w))
	bayer_quad = np.zeros((4, quad_h, quad_w))
	green_grgb = np.zeros((2, quad_h, quad_w))

	bayer_quad[0,:,:] = mosaic[0,0::2,0::2]
	bayer_quad[1,:,:] = mosaic[0,0::2,1::2]
	bayer_quad[2,:,:] = mosaic[0,1::2,0::2]
	bayer_quad[3,:,:] = mosaic[0,1::2,1::2]

	redblue_bayer[0,:,:] = bayer_quad[1,:,:]
	redblue_bayer[1,:,:] = bayer_quad[2,:,:]

	green_grgb[0,:,:] = bayer_quad[0,:,:]
	green_grgb[1,:,:] = bayer_quad[3,:,:]

	bayer_quad = torch.Tensor(bayer_quad)
	redblue_bayer = torch.Tensor(redblue_bayer)
	green_grgb = torch.Tensor(green_grgb)

	return (bayer_quad, redblue_bayer, green_grgb)


def get_weight_file(train_dir, training_info_file, search_models=False):
	max_psnr, best_version, best_epoch = get_model_psnr(train_dir, training_info_file, return_epoch=True)
	#print(f"model {train_dir} using best version {best_version} at epoch {best_epoch}")

	if search_models:
		if best_epoch == 0:
			weight_file = os.path.join(train_dir, f"model_v{best_version}_epoch{best_epoch}_pytorch")
		else:
			weight_file = os.path.join(train_dir, f"model_weights_epoch{best_epoch}")
		# print(weight_file)

	else:
		weight_file = os.path.join(train_dir, f"model_v{best_version}_epoch{best_epoch}_pytorch")

	if not os.path.exists(weight_file):
		weight_file = os.path.join(train_dir, f"model_v{best_version}_pytorch")

	return weight_file, max_psnr


parser = argparse.ArgumentParser()

parser.add_argument("--retraindir", type=str)
parser.add_argument("--searchdir", type=str)

args = parser.parse_args()


retrain_perf = {}
search_perf = {}

for m in os.listdir(args.searchdir):
	train_dir = os.path.join(args.searchdir, m)
	training_info_file = os.path.join(train_dir, f"model_{m}_training_log")
	if not os.path.exists(training_info_file):
		continue
	weight_file, max_psnr = get_weight_file(train_dir, training_info_file, search_models=True)

	search_perf[m] = max_psnr

for m in os.listdir(args.retraindir):
	train_dir = os.path.join(args.retraindir, m)
	training_info_file = os.path.join(train_dir, f"model_{m}_training_log")
	weight_file, max_psnr = get_weight_file(train_dir, training_info_file, search_models=True)

	retrain_perf[m] = max_psnr

for m in retrain_perf:
	retrain_psnr = retrain_perf[m]
	if not m in search_perf:
		print(m)
		continue
	search_psnr = search_perf[m]
	if retrain_psnr < search_psnr:
		print(f"{m} diff {retrain_psnr-search_psnr:.2f} search psnr {search_psnr} retrain_psnr {retrain_psnr}")
		if abs(retrain_psnr-search_psnr) > 0.1:
			with open("redo.txt", "a+") as f:
				f.write(f"{m}\n")			

