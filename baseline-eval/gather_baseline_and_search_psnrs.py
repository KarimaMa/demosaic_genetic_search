import csv
import argparse
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--basedir", type=str, help="directory containing per model psnr info")
parser.add_argument("--outfile", type=str)
parser.add_argument("--search_experiments", type=str, help="search model experiments to include")
parser.add_argument("--xtrans", action="store_true")
parser.add_argument("--superres", action="store_true")
parser.add_argument("--superres_only", action="store_true")

args = parser.parse_args()


if args.xtrans:
	baseline_models = ["markesteijn"]
elif args.superres:
	baseline_models = ["drln+dnet", "FALSR-A+dnet", "FALSR-B+dnet", "FALSR-C+dnet", "prosr+dnet", "rcan+dnet", "tenet"]
	baseline_models += ["espcn+gradhalide", "srcnn+gradhalide", "bicubic+gradhalide"]
	baseline_models += ["espcn+dnet", "srcnn+dnet", "bicubic+dnet"]

elif args.superres_only:
	baseline_models = ["drln", "FALSR-A", "FALSR-B", "FALSR-C", "prosr", "rcan", "raisr", "espcn", "srcnn", "bicubic"]
else:
	baseline_models = ["ahd_median", "lmmse", "vng4", "henz"]

with open(args.outfile, "w") as csvf:
	writer = csv.DictWriter(csvf, fieldnames=["model", "dataset", "psnr"])
	writer.writeheader()

	for m in baseline_models:
		datadir = os.path.join(args.basedir, m)
		psnr_file = os.path.join(datadir, f"{args.dataset}_psnrs.csv")
		with open(psnr_file, "r") as pf:
			reader = csv.DictReader(pf)
			avg_psnr = np.mean([float(r["psnr"]) for r in reader])
			print(datadir, avg_psnr)

		writer.writerow({"model":m, "dataset":args.dataset, "psnr":avg_psnr})

	for experiment in args.search_experiments.split(","):
		search_models_dir = os.path.join(args.basedir, experiment)
		for m in os.listdir(search_models_dir):
			if ".txt" in m or ".csv" in m or ".py" in m:
				continue
			datadir = os.path.join(search_models_dir, m)

			psnr_file = os.path.join(datadir, f"{args.dataset}_psnrs.csv")
			if not os.path.exists(psnr_file):
				continue
			with open(psnr_file, "r") as pf:
				reader = csv.DictReader(pf)
				avg_psnr = np.mean([float(r["psnr"]) for r in reader])
				print(datadir, avg_psnr)

			writer.writerow({"model":f"{experiment}_{m}", "dataset":args.dataset, "psnr":avg_psnr})
