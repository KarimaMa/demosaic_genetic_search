import argparse
import os
import sys
from paretoset import paretoset
import pandas as pd 
import numpy as np
from data_scrape_util import *
import csv


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--search_models", type=str, help="directory of search models")
	parser.add_argument("--outcsv", type=str, help="output csv file")
	args = parser.parse_args()

	model_ids, costs, psnrs, best_inits = collect_model_info(args.search_models)

	with open(args.outcsv, "w") as csvf:
		writer = csv.DictWriter(csvf, fieldnames=["modelid", "cost", "psnr"])
		writer.writeheader()
		for i, model in enumerate(model_ids):
			writer.writerow({"modelid": model, "cost": costs[i], "psnr": psnrs[i]})






	