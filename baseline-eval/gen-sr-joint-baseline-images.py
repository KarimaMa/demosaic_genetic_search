import argparse
import os
import time

import numpy as np
import torch as th
import imageio
import sys
import csv
rootdir = '/'.join(sys.path[0].split("/")[0:-1])
sys.path.append(rootdir)
import dataset_util
import util
from dataset_util import FastDataLoader, tensor2image, im2tensor
from superres_only_dataset import SDataset, SBaselineDataset
import demosaicnet

# import RAISR model
sr_baseline_dir = os.path.join("/".join(sys.path[0].split("/")[0:3]), "sr-baselines")
print(sr_baseline_dir)
sys.path.append(sr_baseline_dir)
from raisr.test import RAISR

import logging
logger = logging.getLogger("PIL.PngImagePlugin")
logger.setLevel("ERROR")

"""
Runs joint SR + demosaicking models
"""

def run(datafile, model, output):
    print(f"Prediction using {model} Torchscript (CPU)")
    ROOT = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(ROOT, model)
    if model == "raisr":
        ts_model = RAISR()
        model_name = model
    else:
        ts_model = th.jit.load(args.model)
        model_name = os.path.splitext(model)[0].split("/")[-1]

    print(f"model name {model_name}")
    
    model_output_dir = os.path.join(output, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Preserve size, use padding
    demosaicker = demosaicnet.BayerDemosaick(pad=True)
   
    image_files = dataset_util.ids_from_file(datafile)

    is_baseline_dataset = not (args.dataset == "hdrvdp" or args.dataset == "moire")

    if is_baseline_dataset:
        dataset = SBaselineDataset(datafile, return_index=True)
    else:
        dataset = SDataset(datafile, return_index=True)

    n = len(dataset)
    indices = list(range(n))

    sampler = th.utils.data.sampler.SequentialSampler(indices)

    data_queue = FastDataLoader(
        dataset, batch_size=1,
        sampler=sampler,
        pin_memory=True, num_workers=1)

    psnr_tracker = util.AverageMeter()

    psnr_dict = {}

    for index, lowres_image, target in data_queue:
        lowres_bayer = demosaicnet.bayer(lowres_image)        
        target = target.squeeze(0)

        image_f = image_files[index]
        subdir, image_id = dataset_util.get_image_id(image_f)
        if is_baseline_dataset:
            image_id = image_id.replace("LR.png", "HR.png")

        outdir = os.path.join(model_output_dir, subdir)
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, image_id)

        start = time.time()

        if args.joint:
            if os.path.basename(model) == "tenet.pt":
                model_input = lowres_bayer.squeeze(0)
                print("TENET model: joint demosaicking + superres")
            else:
                print("Demosaicking with Demosaicnet")
                model_input = demosaicker(lowres_bayer).squeeze(0)
        else:
            assert os.path.basename(model) != "tenet.pt", "cannot run Tenet in SR only mode"
            model_input = lowres_image.squeeze(0)

        if model_name == "raisr":
            model_input = tensor2image(model_input.unsqueeze(0))
        # try:
        print(type(model_input))
        print(model_input.shape)

        out = ts_model(model_input)
        elapsed = (time.time() - start) * 1000

        if model_name == 'raisr':
            out = im2tensor(out)
            print(f"out {out.shape} {out.type}") 

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

        out = tensor2image(th.clamp(out.unsqueeze(0), 0, 1))
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
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--crop", type=int, default=0)

    args = parser.parse_args()
    run(args.datafile, args.model, args.output)
