import argparse
import os
import time

import numpy as np
import torch as th
import tensorflow as tf
import imageio
import scipy
import skimage.color as sc
import skimage
import skimage.transform as xform
from skimage import measure
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
print(rootdir)
from dataset_util import ids_from_file, FastDataLoader
from superres_only_dataset import SBaselineDataset, SDataset
import util

import demosaicnet

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


def run(datafile, model, output, joint):
    images = ids_from_file(datafile)

    print(f"Prediction using {model} Torchscript (CPU)")
    ROOT = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(ROOT, model)
    model_name = os.path.splitext(model)[0].split("/")[-1]

    # model_name = os.path.basename(os.path.splitext(model)[0])
    os.makedirs(os.path.join(output, model_name), exist_ok=True)
    demosaicker = None
    if joint:
        demosaicker = demosaicnet.BayerDemosaick(pad=True)
    else:
        print("Super-res only, no demosaicking")

    print("loading graph")
    with tf.Graph().as_default():
        output_graph_def = tf.compat.v1.GraphDef()
        with open(model, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.compat.v1.Session() as sess:
            y_image = sess.graph.get_tensor_by_name("input_image_evaluate_y:0")
            pbpr_image = sess.graph.get_tensor_by_name("input_image_evaluate_pbpr:0")
            output_tensor = sess.graph.get_tensor_by_name('test_sr_evaluator_i1_b0_g/target:0')
            sess.run(tf.compat.v1.global_variables_initializer())

            model_output_dir = os.path.join(output, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            image_files = ids_from_file(datafile)

            is_baseline_dataset = not (args.dataset == "hdrvdp" or args.dataset == "moire")
            if is_baseline_dataset:
                dataset = SBaselineDataset(datafile, return_index=True)
            else:
                dataset = SDataset(datafile, return_index=True)

            n = len(dataset)
            indices = list(range(n))

            sampler = th.utils.data.sampler.SequentialSampler(indices)

            train_queue = FastDataLoader(
                dataset, batch_size=1,
                sampler=sampler,
                pin_memory=True, num_workers=1)

            psnr_tracker = util.AverageMeter()
            psnr_dict = {}
            
            print("Super-res only, no demosaicking")
            
            for index, lr_img, hr_img in train_queue:
                image_f = image_files[index]
                subdir, image_id = get_image_id(image_f)
                if is_baseline_dataset:
                    image_id = image_id.replace("LR.png", "HR.png")
                outdir = os.path.join(model_output_dir, subdir)
                os.makedirs(outdir, exist_ok=True)
                outfile = os.path.join(outdir, image_id)
              
                if joint:
                    print("Simulating Bayer mosaic")
                    lr_img = demosaicnet.bayer(lr_img)
                    print("Demosaicking with Demosaicnet")
                    lr_img = demosaicker(lr_img)

                lr_img = np.transpose(lr_img.squeeze(0).detach().numpy(), [1,2,0])

                start = time.time()
                ypbpr = sc.rgb2ypbpr(lr_img) # / 255.0)
                size = lr_img.shape

                x_scale = xform.resize(
                    lr_img, [size[0] * 2, size[1] * 2], order=3)
                y = ypbpr[..., 0]
                pbpr = sc.rgb2ypbpr(x_scale)[..., 1:]
                y = np.expand_dims(y, -1)
                paras = {y_image: [y], pbpr_image: [pbpr]}

                out = sess.run(output_tensor, paras)
                out = out[0]

                out = np.clip(out, 0, 1)
                out = th.tensor(np.transpose(out, [2,0,1])).unsqueeze(0)
                print(f"{image_f} target {hr_img.shape} out {out.shape}")

                if args.crop > 0:
                    crop = args.crop
                    out = out[:,:,crop:-crop,crop:-crop]
                    hr_img = hr_img[:,:,crop:-crop,crop:-crop]

                elapsed = (time.time() - start) * 1000
                npix = hr_img.shape[1]*hr_img.shape[2]
                time_per_mpix = elapsed / (npix * 1e-6)
                print(f"Time elapsed {elapsed:.1f} ms ({time_per_mpix:.1f} ms/Megapixel)")

                per_image_mse = (out-hr_img).square().mean(-1).mean(-1).mean(-1).mean(-1)
                per_image_psnr = -10.0*th.log10(per_image_mse)
                print(f"psnr {per_image_psnr.item():.2f}")
                print("------")

                psnr_tracker.update(per_image_psnr.item(), 1)

                psnr_dict[os.path.join(subdir, image_id)] = per_image_psnr.item()

                out = tensor2image(out)
                imageio.imsave(outfile, out)


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
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--crop", type=int, default=0)
    parser.add_argument("--joint", action="store_true")

    args = parser.parse_args()
    run(args.datafile, args.model, args.output, args.joint)

