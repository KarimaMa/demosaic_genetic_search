import csv
import argparse
import os
import sys
sys.path.append(sys.path[0].split("/")[0])
from util import compute_psnr


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--model_versions", type=int)

args = parser.parse_args()

summary_train_csv = os.path.join(args.data_dir, f"train_losses.csv")
summary_train_f = open(summary_train_csv, "w", newline="\n")
summary_train_writer = csv.writer(summary_train_f, delimiter=",")

summary_validation_csv = os.path.join(args.data_dir, f"validation_losses.csv")
summary_validation_f = open(summary_validation_csv, "w", newline="\n")
summary_validation_writer = csv.writer(summary_validation_f, delimiter=",")

summary_validation_psnr_csv = os.path.join(args.data_dir, f"validation_psnr.csv")
summary_validation_psnr_f = open(summary_validation_psnr_csv, "w", newline="\n")
summary_validation_psnr_writer = csv.writer(summary_validation_psnr_f, delimiter=",")

summary_data = {}

for version in range(args.model_versions):
	summary_train_data = {"batch": [], "loss": []}
	summary_val_data = {"batch": [], "loss": []}
	train_log = os.path.join(args.data_dir, f"v{version}_train_log")
	validation_log = os.path.join(args.data_dir, f"v{version}_validation_log")

	with open(train_log, "r") as f:
		for l in f:
			print(l)

			batch = int(l.split(' ')[-2].strip())
			loss = float(l.split(' ')[-1].strip())
			
			summary_train_data["batch"].append(batch)
			summary_train_data["loss"].append(loss)

	with open(validation_log, "r") as f:
		for l in f:
			print(l)

			if "epoch" in l:
				continue

			batch = int(l.split(' ')[-2].strip())
			loss = float(l.split(' ')[-1].strip())

			summary_val_data["batch"].append(batch)
			summary_val_data["loss"].append(loss)

	summary_data[f"v_{version}"] = {"train_data": summary_train_data, "validation_data": summary_val_data}


# write train data for all subsets and versios
header = ["batch"]
for version in range(args.model_versions):
	header += [f"v_{version} loss"] 

summary_train_writer.writerow(header)

n_rows = len(summary_data["v_0"]["train_data"]["batch"])
for row in range(n_rows):
	row_data = [summary_data["v_0"]["train_data"]["batch"][row]]
	for version in range(args.model_versions):
		v = f"v_{version}"
		row_data.append(summary_data[v]["train_data"]["loss"][row])
	summary_train_writer.writerow(row_data)

# write validation data for all subsets and versions
summary_validation_writer.writerow(header)

n_rows = len(summary_data["v_0"]["validation_data"]["batch"])
for row in range(n_rows):
	row_data = [summary_data["v_0"]["validation_data"]["batch"][row]]
	for version in range(args.model_versions):
		v = f"v_{version}"
		row_data.append(summary_data[v]["validation_data"]["loss"][row])
	summary_validation_writer.writerow(row_data)

# write psnrs
header = ["batch"]
for version in range(args.model_versions):
	header += [f"v_{version} PSNR"] 

summary_validation_psnr_writer.writerow(header)

for row in range(n_rows):
	row_data = [summary_data["v_0"]["validation_data"]["batch"][row]]
	for version in range(args.model_versions):
		v = f"v_{version}"
		loss = summary_data[v]["validation_data"]["loss"][row]
		row_data += [compute_psnr(loss)]

	summary_validation_psnr_writer.writerow(row_data)

