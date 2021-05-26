import csv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--basedir", type=str, help="directory containing per model psnr info")
parser.add_argument("--outfile", type=str)
parser.add_argument("--search_experiments", type=str, help="search model experiments to include")
parser.add_argument("--xtrans", action="store_true")

args = parser.parse_args()


if args.xtrans:
	baseline_models = ["markesteijn"]
else:
	baseline_models = ["ahd_median", "lmmse", "vng4", "henz"]

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

	for experiment in args.search_experiments.split(","):
		search_models_dir = os.path.join(args.basedir, experiment)
		for m in os.listdir(search_models_dir):
			if ".txt" in m or ".csv" in m or ".py" in m:
				continue
			datadir = os.path.join(search_models_dir, m)

			psnr_file = os.path.join(datadir, f"{args.dataset}_avg_psnr.txt")
			if not os.path.exists(psnr_file):
				continue
			print(psnr_file)
			psnr = [float(l) for l in open(psnr_file, "r")][0]
			print(psnr)
			writer.writerow({"model":f"{experiment}_{m}", "dataset":args.dataset, "psnr":psnr})