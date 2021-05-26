import csv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--basedir", type=str, "directory containing per model psnr info")
parser.add_argument("--outfile", type=str)
parser.add_argument("--xtrans", action="store_true")
parser.add_argument("--superres", action="store_true")

args = parser.parse_args()

if args.xtrans:
	baseline_models = ["markesteijn"]
elif args.superres:
	baseline_models = ["rcan", "drln", "prosr", "tenet"]
else:
	baseline_models = ["ahd_median", "lmmse", "vgn4"]

with open(args.outfile, "w") as csvf:
	writer = csv.DictWriter(csvf, fieldnames=["model", "dataset", "psnr"])
	writer.writeheader()

	for m in baseline_models:
		datadir = os.path.join(args.basedir, m)
		psnr_file = os.path.join(datadir, f"{args.dataset}_avg_psnr.txt")
		print(psnr_file)

		psnr = [float(l) for l in open(psnr_file, "r")][0]
		print(psnr)
		writer.writerow({"model":m, "dataset":args.dataset, "psnr":psnr})

	search_models_dir = os.path.join(args.basedir, "SEARCH_MODELS")
	for m in os.listdir(search_models_dir):
		datadir = os.path.join(search_models_dir, m)
		psnr_file = os.path.join(datadir, f"{args.dataset}_avg_psnr.txt")
		print(psnr_file)
		psnr = [float(l) for l in open(psnr_file, "r")][0]
		print(psnr)
		writer.writerow({"model":f"search_{m}", "dataset":args.dataset, "psnr":psnr})