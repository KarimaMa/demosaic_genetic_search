import os
import csv
import argparse


"""
returns the best PSNR obtained for a given initialization
"""
def get_psnr(file):
	with open(file, "r") as f:
		psnrs = [float(l.strip().split(' ')[-1]) for l in f]
		if len(psnrs) == 0:
			return None
		return max(psnrs)


def get_psnr_delta(model_dir, inits):
	psnrs = []
	for i in range(inits):
		file = os.path.join(model_dir, f"v{i}_validation_log")
		if not os.path.exists(file):
			return None

		psnr_i = get_psnr(file)
		if not psnr_i is None:
			psnrs.append(psnr_i)

	if len(psnrs) != inits:
		return None

	return max(psnrs) - min(psnrs)


def write_csv(model_rootdir, inits, outfile):
	with open(outfile, "w") as f:
		fieldnames = ['modelid', 'psnr_delta']
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()

		for modelid in os.listdir(model_rootdir):
			modeldir = os.path.join(model_rootdir, modelid)
			psnr_delta = get_psnr_delta(modeldir, inits)
			if psnr_delta is None:
				continue
			writer.writerow({'modelid': modelid, 'psnr_delta': psnr_delta})


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_root", type=str)
	parser.add_argument("--inits", type=int, default=3)
	parser.add_argument("--outfile", type=str)
	args = parser.parse_args()

	write_csv(args.model_root, args.inits, args.outfile)