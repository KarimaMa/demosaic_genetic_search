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
from mosaic_gen import xtrans, xtrans_cell, xtrans_3x3_invariant


import logging
logger = logging.getLogger("PIL.PngImagePlugin")
logger.setLevel("ERROR")


def tensor2image(t, normalize=False, dtype=np.uint8):
    """Converts an tensor image (4D tensor) to a numpy 8-bit array.

    Args:
        t(th.Tensor): input tensor with dimensions [bs, c, h, w], c=3, bs=1
        normalize(bool): if True, normalize the tensor's range to [0, 1] before
            clipping
    Returns:
        (np.array): [h, w, c] image in uint8 format, with c=3
    """
    assert len(t.shape) == 4, "expected 4D tensor, got %d dimensions" % len(t.shape)
    bs, c, h, w = t.shape

    assert bs == 1, "expected batch_size 1 tensor, got %d" % bs
    t = t.squeeze(0)

    assert c == 3 or c == 1, "expected tensor with 1 or 3 channels, got %d" % c

    if normalize:
        m = t.min()
        M = t.max()
        t = (t-m) / (M-m+1e-8)

    t = torch.clamp(t.permute(1, 2, 0), 0, 1).cpu().detach().numpy()

    if dtype == np.uint8:
        return (255.0*t).astype(np.uint8)
    elif dtype == np.uint16:
        return ((2**16-1)*t).astype(np.uint16)
    else:
        raise ValueError("dtype %s not recognized" % dtype)


def im2tensor(im):
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    elif im.dtype == np.uint16:
        im = im.astype(np.float32) / (2**16-1.0)
    else:
        raise ValueError(f"unknown input type {im.dtype}")
    im = torch.from_numpy(im)
    if len(im.shape) == 2:  # grayscale -> rgb
        im = im.unsqueeze(-1).repeat(1, 1, 3)
    im = im.float().permute(2, 0, 1)
    return im

def get_image_id(image_f):
    subdir = "/".join(image_f.split("/")[-3:-1])
    image_id = image_f.split("/")[-1]
    return subdir, image_id


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


def get_largest_multiple(value, factor):
	for i in range(value, 0, -1):
		if i % factor == 0 and i % 2 == 0 and i % 4 == 0:
			return i

def gen_xtrans_inputs(image_f):
	img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
	img = np.transpose(img, [2, 0, 1])
	oldh = img.shape[-2]
	oldw = img.shape[-1]

	h = get_largest_multiple(oldh, 6)
	w = get_largest_multiple(oldw, 6)
	hc = (oldh - h)//2
	wc = (oldw - w)//2
	if hc != 0:
		img = img[:,hc:-hc,:]
	if wc != 0:
		img = img[:,:,wc:-wc]

	mask = xtrans_cell(h=img.shape[-2], w=img.shape[-1])

	mosaic3chan = xtrans(img, mask=mask)
	flat_mosaic = np.sum(mosaic3chan, axis=0, keepdims=True)
	packed_mosaic = xtrans_3x3_invariant(flat_mosaic)

	# extract out the RB values from the mosaic
	period = 6
	num_blocks = 4
	rb_shape = list(flat_mosaic.shape)
	rb_shape[0] = 16
	rb_shape[1] //= period
	rb_shape[2] //= period

	rb = np.zeros(rb_shape, dtype=np.float32)
	num_blocks = 4
	block_size = 4 # 4 red and blue values per 3x3

	for b in range(num_blocks):
	  for i in range(block_size):
	    bx = b % 2
	    by = b // 2
	    x = bx * 3 + (i*2+1) % 3
	    y = by * 3 + (i*2+1) // 3
	    c = b * block_size + i
	    rb[c, :, :] = flat_mosaic[0, y::period, x::period]

	input = torch.Tensor(packed_mosaic), torch.Tensor(mosaic3chan), torch.Tensor(flat_mosaic), torch.Tensor(rb)

	target = img
	target = torch.Tensor(target)

	return (input, target) 


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

parser.add_argument("--xtrans", action="store_true")
parser.add_argument("--gradienthalide", action="store_true")
parser.add_argument("--demosaicnet", action="store_true")
parser.add_argument("--search_models", action="store_true")
parser.add_argument("--nas_models", action="store_true")
parser.add_argument("--dnetgrid_models", action="store_true")

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

	training_info_file = os.path.join(train_dir, f"model_{args.model}_training_log")
	weight_file, max_psnr = get_weight_file(train_dir, training_info_file, search_models=(args.search_models or args.nas_models or args.dnetgrid_models))

	# see if search produced better psnrs than retraining
	is_search_model = (args.search_models or args.nas_models)
	if is_search_model:
		train_dir = os.path.join(args.searchdir, args.model)
		training_info_file = os.path.join(train_dir, f"model_{args.model}_training_log")
		search_weight_file, search_max_psnr = get_weight_file(train_dir, training_info_file, search_models=True)
		if search_max_psnr > max_psnr:
			with open("retrain-v-search.txt", "a+") as f:
				f.write(f"{args.model} search psnr {search_max_psnr} retrain psnr {max_psnr}\n")
			weight_file = search_weight_file

	with open("check.txt", "a+") as f:
		f.write(f"{weight_file}\n")

	if args.demosaicnet:
		params = args.model.split("_")[-1]
		depth = int(params.split("W")[0][1:])
		width = int(params.split("W")[1])
		print(f"eval {args.model} depth {depth} width {width}")
		model = FullDemosaicknet(depth=depth, width=width)
		model.load_state_dict(torch.load(weight_file))
		model.eval()

	elif args.dnetgrid_models:
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

		if args.search_models:
			args.green_model_ast_files = [l.strip() for l in open(args.green_model_asts)]
			args.green_model_weight_files = [l.strip() for l in open(args.green_model_weights)]
			insert_green_model(model, args.green_model_ast_files, args.green_model_weight_files)

		print(model.dump())

		model = model.ast_to_model()
		model.load_state_dict(torch.load(weight_file))

	perf_data = {"image":[], "psnr":[]}

	with torch.no_grad():
		for f in os.listdir(args.eval_imagedir):
			if not f.endswith(".png"):
				continue
			print(f"gt {f}")

			ground_truth_f = os.path.join(args.eval_imagedir, f)
			img = np.array(imread(ground_truth_f)).astype(np.float32) / (2**8-1)
			img = torch.tensor(np.transpose(img, [2,0,1]))

			bayer = demosaicnet.bayer(img).unsqueeze(0)

			if args.gradienthalide:
				flatbayer = torch.sum(bayer, axis=1, keepdims=True)
				output = model((flatbayer, bayer))
			elif args.demosaicnet or args.dnetgrid_models:
				output = model(bayer)
			else:
				if args.search_models:
					if args.xtrans:
						(packed_mosaic, mosaic3chan, flat_mosaic, rb), img = gen_xtrans_inputs(ground_truth_f)
						# print(f"img {img.shape} mosaic3chan {mosaic3chan.shape}")
						model_inputs = {"Input(Mosaic)": mosaic3chan.unsqueeze(0),
							"Input(Mosaic3x3)": packed_mosaic.unsqueeze(0), 
							"Input(FlatMosaic)": flat_mosaic.unsqueeze(0),
							"Input(RBXtrans)": rb.unsqueeze(0)}        
					else:
						bayer_quad, redblue_bayer, green_grgb = gen_model_inputs(bayer.squeeze(0))
						model_inputs = {"Input(Mosaic)": bayer_quad.unsqueeze(0), 
							"Input(Green@GrGb)": green_grgb.unsqueeze(0), 
							"Input(RedBlueBayer)": redblue_bayer.unsqueeze(0)}
				else:
					model_inputs = {"Input(Mosaic)": bayer}

				model.reset()
				output = model.run(model_inputs)

			clamped = torch.clamp(output, 0, 1)

			# can't afford to save all pareto model outputs
			if (not args.search_models) and (not args.nas_models):
				out_img = tensor2image(clamped)
				outfile = os.path.join(model_outputdir, f)
				print(outfile)
				imageio.imsave(outfile, out_img)

			clamped = clamped.squeeze(0)

			if args.crop > 0:
				crop = args.crop
				clamped = clamped[:,crop:-crop,crop:-crop]
				img = img[:,crop:-crop,crop:-crop]

			# print(f"after crop img {img.shape} out {clamped.shape}")

			image_mse = (clamped-img).square().mean(-1).mean(-1).mean(-1)
			image_psnr = -10.0*torch.log10(image_mse)

			perf_data["image"].append(f)
			perf_data["psnr"].append(image_psnr.item())
	     
			print(f"psnr: {image_psnr}")

			subdir, image_id = get_image_id(f)
			print(f"image {image_id}")
			outdir = os.path.join(model_outputdir, subdir)
			os.makedirs(outdir, exist_ok=True)
			outfile = os.path.join(outdir, image_id)
			print(outfile)

			out = tensor2image(torch.clamp(clamped.unsqueeze(0), 0, 1))
			imageio.imsave(outfile, out)

else:
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


