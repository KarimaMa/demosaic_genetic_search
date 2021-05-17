import os
import numpy as np
from imageio import imread
from config import IMG_H, IMG_W
from util import ids_from_file
from mosaic_gen import bayer, xtrans, xtrans_3x3_invariant, xtrans_cell
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--datafile", type=str)
	parser.add_argument("--outdir", type=str)
	parser.add_argument("--xtrans", action="store_true")

	args = parser.parse_args()
	image_files = ids_from_file(args.datafile) # patch filenames


	for image_f in image_files:
		img = np.array(imread(image_f)).astype(np.float32) / (2**8-1)
		img = np.transpose(img, [2, 0, 1])
		img = img[:,4:-4,4:-4]

		if args.xtrans:
			mosaic3chan = xtrans(img)
			mosaic = np.sum(mosaic3chan, axis=0, keepdims=True)

		else:
			mosaic = bayer(img)
			mosaic = np.sum(mosaic, axis=0, keepdims=True)

		subdir = "/".join(image_f.split("/")[-4:-1])
		outdir = os.path.join(args.outdir, subdir)
		if not os.path.exists(outdir):
			os.makedirs(outdir)

		image_id = image_f.split("/")[-1].replace(".png", ".mosaic")
		outfile = os.path.join(outdir, image_id)
		print(outfile)
		np.save(outfile, mosaic)
