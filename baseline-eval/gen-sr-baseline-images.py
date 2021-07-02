import argparse
import os
import time
import numpy as np
import torch as th
import imageio
import sys
import csv
from PIL import Image
rootdir = '/'.join(sys.path[0].split("/")[0:-1])
sys.path.append(rootdir)
import dataset_util
import util
from dataset_util import FastDataLoader, tensor2image, im2tensor
from superres_only_dataset import SDataset, SBaselineDataset
import demosaicnet
from gradienthalide_models import GradientHalide

# import RAISR model
sr_baseline_dir = os.path.join("/".join(sys.path[0].split("/")[0:3]), "sr-baselines")
print(sr_baseline_dir)
sys.path.append(sr_baseline_dir)
from raisr.test import RAISR

# import SRCNN model
from srcnn.models import SRCNN
from srcnn.utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb

from upsample import BicubicUpsample

import logging
logger = logging.getLogger("PIL.PngImagePlugin")
logger.setLevel("ERROR")

"""
Runs joint SR + demosaicking models
"""

def run(datafile, model, output, args):
    print(f"Prediction using {model} Torchscript (CPU)")
    ROOT = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(ROOT, model)
    if model == "raisr":
        ts_model = RAISR()
        model_name = model
    elif model == "srcnn":
        ts_model = SRCNN()
        model_name = model
        ts_model.load_state_dict(th.load(args.weights))
        ts_model.eval()
        bicubic_upsampler = BicubicUpsample(3)
    elif model == "bicubic":
        ts_model = BicubicUpsample(3)
        model_name = model
    else:
        ts_model = th.jit.load(args.model)
        model_name = os.path.splitext(model)[0].split("/")[-1]
        ts_model.eval()
    
    if args.joint:
        if args.cheap_demosaicker:
            model_output_dir = os.path.join(output, model_name + "+gradhalide")
        elif not model_name == "tenet":
            model_output_dir = os.path.join(output, model_name + "+dnet")
    else:
        model_output_dir = os.path.join(output, model_name)

    os.makedirs(model_output_dir, exist_ok=True)

    # Preserve size, use padding
    # use cheap demosaicker for cheap models
    if args.joint:
        if args.cheap_demosaicker:
            demosaicker = GradientHalide(7, 15)
            weight_file = "/home/karima/bayer-baselines/GRADHALIDE_BASELINES/MODEL-K7F15/model_train/model_v1_epoch2_pytorch"
            demosaicker.load_state_dict(th.load(weight_file))
        else:
            demosaicker = demosaicnet.BayerDemosaick(pad=True)
        demosaicker.eval()

    image_files = dataset_util.ids_from_file(datafile)

    is_baseline_dataset = not args.dataset in ["hdrvdp","moire","SRTest","kodak","mcm"]

    if is_baseline_dataset:
        dataset = SBaselineDataset(datafile, return_index=True)
    else:
        dataset = SDataset(datafile, return_index=True, crop_file=args.crop_file)

    n = len(dataset)
    indices = list(range(n))

    sampler = th.utils.data.sampler.SequentialSampler(indices)

    data_queue = FastDataLoader(
        dataset, batch_size=1,
        sampler=sampler,
        pin_memory=True, num_workers=1)

    psnr_tracker = util.AverageMeter()

    psnr_dict = {}

    with th.no_grad():
        for index, lowres_image, target in data_queue:
            lowres_bayer = demosaicnet.bayer(lowres_image)        
            target = target.squeeze(0)

            image_f = image_files[index]
            _, image_id = dataset_util.get_image_id(image_f)
            subdir = args.dataset
            if is_baseline_dataset:
                image_id = image_id.replace("LR.png", "HR.png")

            outdir = os.path.join(model_output_dir, subdir)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, image_id)
            print(outfile)
            exit()
            
            start = time.time()

            if args.joint:
                if os.path.basename(model) == "tenet.pt":
                    model_input = lowres_bayer.squeeze(0)
                    print("TENET model: joint demosaicking + superres")
                else:
                    if args.cheap_demosaicker:
                        print("Demosaicking with GradientHalide")
                        flat_lowres_bayer = th.sum(lowres_bayer, axis=1, keepdims=True)
                        model_input = demosaicker((flat_lowres_bayer, lowres_bayer)).squeeze(0).clamp(0.0,1.0)
                    else:
                        print("Demosaicking with Demosaicnet")
                        model_input = demosaicker(lowres_bayer).squeeze(0)
            else:
                assert os.path.basename(model) != "tenet.pt", "cannot run Tenet in SR only mode"
                model_input = lowres_image.squeeze(0)

            if model_name == "raisr":
                model_input = tensor2image(model_input.unsqueeze(0))
                
            elif model_name == "srcnn":
                # img = np.transpose(model_input.squeeze(0).numpy(), [1,2,0]) * 255.0
                # h,w,c = img.shape

                # img = Image.fromarray(img.astype(np.uint8))
                # img = img.convert("YCbCr")
                # upsampled = np.array(img.resize((w*2, h*2), resample=Image.BICUBIC))

                # Y, Cb, Cr = img.split()
                # Y = th.from_numpy(np.array(Y)).unsqueeze(0).float() / 255.0
                model_input = bicubic_upsampler(model_input).unsqueeze(0)
                # ycbcr = convert_rgb_to_ycbcr(upsampled)
                # y = ycbcr[..., 0] / 255.0
                # model_input = th.from_numpy(y.astype(np.float32)).unsqueeze(0).unsqueeze(0)

            # try:
            out = ts_model(model_input)
            elapsed = (time.time() - start) * 1000

            if model_name == 'raisr':
                out = im2tensor(out)
            elif model_name == "srcnn":
                out = out[0]
            #     out = out.clamp(0.0,1.0).mul(255.0).numpy().squeeze(0).squeeze(0)
            #     out = np.array([out, upsampled[..., 1], upsampled[..., 2]]).transpose([1, 2, 0])
            #     out = th.Tensor(np.transpose(convert_ycbcr_to_rgb(out), [2,0,1])) / 255.0

            npix = target.shape[1]*target.shape[2]
            time_per_mpix = elapsed / (npix * 1e-6)
            print(f"Time elapsed {elapsed:.1f} ms ({time_per_mpix:.1f} ms/Megapixel)")

            clamped = th.clamp(out, min=0, max=1)

            if args.crop > 0:
                crop = args.crop
                clamped = clamped[:,crop:-crop,crop:-crop]
                target = target[:,crop:-crop,crop:-crop]
        
            per_image_mse = (clamped-target).square().mean(-1).mean(-1).mean(-1)
            per_image_psnr = -10.0*th.log10(per_image_mse)
            print(f"psnr {per_image_psnr.item():.2f}")

            psnr_tracker.update(per_image_psnr.item(), 1)

            psnr_dict[os.path.join(subdir, image_id)] = per_image_psnr.item()

            print(model_input.shape, out.shape, clamped.shape)
            out = tensor2image(clamped.unsqueeze(0))
            imageio.imsave(outfile, out)
            # except Exception as e:
            #     with open("failed_sr_baseline_models.txt", "a+") as f:
            #         f.write(f"{model_name} {args.dataset}")
            #         f.write(f"{e}\n")

    with open(os.path.join(model_output_dir, f"{args.dataset}_psnrs.csv"), "w") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["image", "psnr"])
        writer.writeheader()
        for imagename in psnr_dict:
            writer.writerow({"image":imagename, "psnr": psnr_dict[imagename]})

    with open(os.path.join(model_output_dir, f"{args.dataset}_avg_psnr.txt"), "w") as f:
        f.write(f"{psnr_tracker.avg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str)
    parser.add_argument("--output", default="output")
    parser.add_argument("--model", default="drln.pt")
    parser.add_argument("--weights", type=str, help="needed for srcnn, srcnn uses weight file not model file")
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--crop", type=int, default=0)
    parser.add_argument("--crop_file", type=str, help="saving cropping information")
    parser.add_argument("--cheap_demosaicker", action="store_true")

    args = parser.parse_args()
    run(args.datafile, args.model, args.output, args)
