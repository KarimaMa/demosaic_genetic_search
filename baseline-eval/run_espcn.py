import argparse
import os
import time

import numpy as np
import torch as th
import imageio
import scipy
import skimage.color as sc
import skimage
import skimage.transform as xform
from skimage import measure
import argparse
import os
import time
from PIL import Image

import numpy as np
import torch as th
import torch.nn as nn
import imageio
import sys
import csv
rootdir = '/'.join(sys.path[0].split("/")[0:-1])
sys.path.append(rootdir)

sr_baseline_dir = os.path.join("/".join(sys.path[0].split("/")[0:3]), "sr-baselines")
sys.path.append(sr_baseline_dir)
from upsample import BicubicUpsample

from dataset_util import ids_from_file, FastDataLoader
from superres_only_dataset import SBaselineDataset, SDataset
import util

import demosaicnet
from gradienthalide_models import GradientHalide

import logging
logger = logging.getLogger("PIL.PngImagePlugin")
logger.setLevel("ERROR")

from espcn import ESPCN


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

    t = th.clamp(t.permute(1, 2, 0), 0, 1).cpu().detach().numpy()

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
    im = th.from_numpy(im)
    if len(im.shape) == 2:  # grayscale -> rgb
        im = im.unsqueeze(-1).repeat(1, 1, 3)
    im = im.float().permute(2, 0, 1)
    return im

def get_image_id(image_f):
    subdir = "/".join(image_f.split("/")[-3:-1])
    image_id = image_f.split("/")[-1]
    return subdir, image_id


def run(datafile, chkpt, output, joint, cheap_demosaicker):
    print(f"Prediction using ESPCN (CPU)")

    model = ESPCN(2)
    upsampler = BicubicUpsample(1)

    state = th.load(chkpt, map_location="cpu")

    print(f"Loading model weights: {chkpt}")
    model.load_state_dict(state, strict=True)

    model_name = "espcn"

    # Preserve size, use padding
    demosaicker = None
    if joint:
        if cheap_demosaicker:
            model_output_dir = os.path.join(output, model_name + "+gradhalide")
            demosaicker = GradientHalide(7, 15)
            weight_file = "/home/karima/bayer-baselines/GRADHALIDE_BASELINES/MODEL-K7F15/model_train/model_v1_epoch2_pytorch"
            demosaicker.load_state_dict(th.load(weight_file))
        else:
            model_output_dir = os.path.join(output, model_name + "+dnet")
            demosaicker = demosaicnet.BayerDemosaick(pad=True)
        demosaicker.eval()
    else:
        print("Super-res only, no demosaicking")
        model_output_dir = os.path.join(output, model_name)

    os.makedirs(model_output_dir, exist_ok=True)

    image_files = ids_from_file(datafile)

    is_baseline_dataset = not args.dataset in ["hdrvdp","moire","kodak","mcm"]
    if is_baseline_dataset:
        dataset = SBaselineDataset(datafile, return_index=True)
    else:
        dataset = SDataset(datafile, return_index=True)

    indices = list(range(len(dataset)))

    sampler = th.utils.data.sampler.SequentialSampler(indices)

    train_queue = FastDataLoader(
        dataset, batch_size=1,
        sampler=sampler,
        pin_memory=True, num_workers=1)

    psnr_tracker = util.AverageMeter()
    psnr_dict = {}

    for index, lr_img, hr_img in train_queue:
        image_f = image_files[index]
        subdir, image_id = get_image_id(image_f)
        if is_baseline_dataset:
            image_id = image_id.replace("LR.png", "HR.png")

        outdir = os.path.join(model_output_dir, args.dataset)
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, image_id)

        # Apply mosaic + demosaick depending on the baseline you want to use
        start = time.time()
        if joint:
            lr_bayer = demosaicnet.bayer(lr_img)
            if args.cheap_demosaicker:
                print("Demosaicking with GradientHalide")
                flat_lowres_bayer = th.sum(lr_bayer, axis=1, keepdims=True)
                im = demosaicker((flat_lowres_bayer, lr_bayer))
            else:
                print("Demosaicking with Demosaicnet")
                im = demosaicker(lr_bayer)
            im = tensor2image(im)
        else:
            im = lr_img
            im = tensor2image(im)
            print("Super-res only, no demosaicking")

        npix = hr_img.shape[1]*hr_img.shape[2]
        print("Processing", image_f, "with hr shape", hr_img.shape)

        im = Image.fromarray(im)

        # Processes only luma
        im = im.convert("YCbCr")
        Y, Cb, Cr = im.split()

        Y = th.from_numpy(np.array(Y)).unsqueeze(0).unsqueeze(0).float()

        out_Y = model(Y/255.0)
        out_Y = out_Y*255.0
        out_Y = th.clamp(out_Y, 0, 255).detach().numpy()
        out_Y = Image.fromarray(np.uint8(out_Y[0,0]), mode="L")

        # upsample chroma with bicubic
        # out_Y = Y.resize((hr_img.shape[3], hr_img.shape[2]), Image.BICUBIC)
        Cb = Cb.resize(out_Y.size, Image.BICUBIC)
        Cr = Cr.resize(out_Y.size, Image.BICUBIC)
        # Cb = upsampler(th.from_numpy(np.array(Cb)).float().unsqueeze(0)/255.0) * 255.0
        # Cr = upsampler(th.from_numpy(np.array(Cr)).float().unsqueeze(0)/255.0) * 255.0

        # Cb = th.clamp(Cb, 0, 255).detach().numpy()
        # Cr = th.clamp(Cr, 0, 255).detach().numpy()
        # Cb = Image.fromarray(np.uint8(Cb[0]), mode="L")
        # Cr = Image.fromarray(np.uint8(Cr[0]), mode="L")

        out = Image.merge('YCbCr', [out_Y, Cb, Cr]).convert("RGB")
        out = np.array(out)

        out_tensor = th.Tensor(np.transpose(out, [2, 0, 1])).unsqueeze(0) / 255.0

        if args.crop > 0:
            crop = args.crop
            out_tensor = out_tensor[:,:,crop:-crop,crop:-crop]
            hr_img = hr_img[:,:,crop:-crop,crop:-crop]

        print(out_tensor.shape, hr_img.shape)

        per_image_mse = (out_tensor-hr_img).square().mean(-1).mean(-1).mean(-1).mean(-1)
        per_image_psnr = -10.0*th.log10(per_image_mse)
        psnr_tracker.update(per_image_psnr.item(), 1)
        psnr_dict[os.path.join(subdir, image_id)] = per_image_psnr.item()

        print(f"psnr {per_image_psnr.item():.2f}")
        print("------")

        out = tensor2image(out_tensor)
        imageio.imsave(outfile, out)

        elapsed = (time.time() - start) * 1000
        time_per_mpix = elapsed / (npix * 1e-6)
        print(f"Time elapsed {elapsed:.1f} ms ({time_per_mpix:.1f} ms/Megapixel)")


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
    parser.add_argument("--model", help="weights", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--crop", type=int, default=0)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--cheap_demosaicker", action="store_true")
    args = parser.parse_args()
    run(args.datafile, args.model, args.output, args.joint, args.cheap_demosaicker)

