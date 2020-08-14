import csv
import argparse
import os
import sys
sys.path.append(sys.path[0].split("/")[0])
from util import compute_psnr


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--subsets", type=int)
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

for subset in range(args.subsets):
	summary_data[f"subset_{subset}"] = {}

	model_dir = os.path.join(args.data_dir, f"subset_{subset}")
	
	for version in range(args.model_versions):
		summary_train_data = {"iter": [], "loss": []}
		summary_val_data = {"epoch": [], "loss": []}

		train_log = os.path.join(model_dir, f"v{version}_train_log")
		train_csv = os.path.join(model_dir, f"v{version}_train.csv")
		train_f = open(train_csv, 'w', newline='\n')
		train_writer = csv.writer(train_f, delimiter=',')

		validation_log = os.path.join(model_dir, f"v{version}_validation_log")
		validation_csv = os.path.join(model_dir, f"v{version}_validation.csv")
		validation_f = open(validation_csv, 'w', newline='\n')
		validation_writer = csv.writer(validation_f, delimiter=',')

		with open(train_log, "r") as f:
			for l in f:
				print(l)

				train_iter = int(l.split(' ')[4].strip())
				loss = float(l.split(' ')[5].strip())

				#print(f"train iter {train_iter} loss {loss}")
				
				train_writer.writerow([train_iter, loss])  

				summary_train_data["iter"].append(train_iter)
				summary_train_data["loss"].append(loss)

		with open(validation_log, "r") as f:
			for l in f:
				print(l)

				if not "epoch" in l:
					continue
				epoch = int(l.split(' ')[5].strip())
				loss = float(l.split(' ')[6].strip())

				print(f"epoch {epoch} loss {loss}")

				validation_writer.writerow([epoch, loss])

				summary_val_data["epoch"].append(epoch)
				summary_val_data["loss"].append(loss)

		summary_data[f"subset_{subset}"][f"v_{version}"] = \
			{"train_data": summary_train_data, "validation_data": summary_val_data}

# write train data for all subsets and versios

n_rows = len(summary_data["subset_0"]["v_0"]["train_data"]["iter"])
header = []
for subset in range(args.subsets):
	for version in range(args.model_versions):
		#colnames = [f"s_{subset} v_{version} iter", f"s_{subset} v_{version} loss"] 
		colnames = [f"s_{subset} v_{version} loss"] 
		header += colnames

summary_train_writer.writerow(header)

for row in range(n_rows):
	row_data = []
	for subset in range(args.subsets):
		for version in range(args.model_versions):
			s = f"subset_{subset}"
			v = f"v_{version}"
			#row_data.append(summary_data[s][v]["train_data"]["iter"][row])
			#row_data.append(row)
			row_data.append(summary_data[s][v]["train_data"]["loss"][row])
	summary_train_writer.writerow(row_data)

# write validation data for all subsets and versions

n_rows = len(summary_data["subset_0"]["v_0"]["validation_data"]["epoch"])
header = []
for subset in range(args.subsets):
	colnames = ["epoch"]
	for version in range(args.model_versions):
		#colnames = [f"s_{subset} v_{version} epoch", f"s_{subset} v_{version} loss"] 
		colnames += [f"s_{subset} v_{version} loss"] 
	colnames += [""]	
	header += colnames

summary_validation_writer.writerow(header)

for row in range(n_rows):
	row_data = []
	for subset in range(args.subsets):
		row_data += [f"{row}"]
		for version in range(args.model_versions):
			s = f"subset_{subset}"
			v = f"v_{version}"
			#row_data.append(summary_data[s][v]["validation_data"]["epoch"][row])
			row_data.append(summary_data[s][v]["validation_data"]["loss"][row])
		row_data += ['']
	summary_validation_writer.writerow(row_data)

# write psnrs
header = []
for subset in range(args.subsets):
	colnames = ["epoch"]
	for version in range(args.model_versions):
		#colnames = [f"s_{subset} v_{version} epoch", f"s_{subset} v_{version} loss"] 
		colnames += [f"s_{subset} v_{version} PSNR"] 
	colnames += [""]	
	header += colnames

summary_validation_psnr_writer.writerow(header)

for row in range(n_rows):
	row_data = []
	for subset in range(args.subsets):
		row_data += [f"{row}"]
		for version in range(args.model_versions):
			s = f"subset_{subset}"
			v = f"v_{version}"

			loss = summary_data[s][v]["validation_data"]["loss"][row]
			row_data += [compute_psnr(loss)]
		row_data += ['']

	summary_validation_psnr_writer.writerow(row_data)

