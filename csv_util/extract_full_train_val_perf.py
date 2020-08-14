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

colnames = ["batch"]
for version in range(args.model_versions):
	#colnames = [f"s_{subset} v_{version} epoch", f"s_{subset} v_{version} loss"] 
	colnames += [f"v_{version} loss"] 

train_csv = os.path.join(args.data_dir, f"train.csv")
if os.path.exists(train_csv):
	os.remove(train_csv)
train_f = open(train_csv, 'w', newline='\n')
train_writer = csv.writer(train_f, delimiter=',')
train_writer.writerow(colnames)

colnames = ["epoch"]
for version in range(args.model_versions):
	#colnames = [f"s_{subset} v_{version} epoch", f"s_{subset} v_{version} loss"] 
	colnames += [f"v_{version} loss"] 

validation_csv = os.path.join(args.data_dir, f"validation.csv")
if os.path.exists(validation_csv):
	os.remove(validation_csv)
validation_f = open(validation_csv, 'w', newline='\n')
validation_writer = csv.writer(validation_f, delimiter=',')
validation_writer.writerow(colnames)

validation_psnr_csv = os.path.join(args.data_dir, f"validation_psnr.csv")
if os.path.exists(validation_psnr_csv):
	os.remove(validation_psnr_csv)
validation_psnr_f = open(validation_psnr_csv, 'w', newline='\n')
validation_psnr_writer = csv.writer(validation_psnr_f, delimiter=',')
validation_psnr_writer.writerow(colnames)

train_losses = {}
validation_losses = {}
validation_psnrs = {}

validation_epochs = []
train_iters = []

for version in range(args.model_versions):
	train_log = os.path.join(args.data_dir, f"v{version}_train_log")
	validation_log = os.path.join(args.data_dir, f"v{version}_validation_log")
	
	version_train_loss = []

	with open(train_log, "r") as f:
		for l in f:
			train_iter = int(l.split(' ')[4].strip())
			loss = float(l.split(' ')[5].strip())

			print(l)
			print(f"train iter {train_iter} loss {loss}")
			
			if version == 0:
				train_iters += [train_iter]

			version_train_loss += [loss]


			train_writer.writerow([train_iter, loss])  

	train_losses[f"v_{version}"] = version_train_loss
	
	version_valid_loss = []
	version_valid_psnr = []

	with open(validation_log, "r") as f:
		for l in f:
			if not "epoch" in l:
				continue
			epoch = int(l.split(' ')[5].strip())
			loss = float(l.split(' ')[-1].strip())
			print(l)
			print(f"epoch {epoch} loss {loss}")

			if version == 0:
				validation_epochs += [epoch]

			version_valid_loss += [loss]
			version_valid_psnr += [compute_psnr(loss)]

			validation_writer.writerow([epoch, loss])

	validation_losses[f"v_{version}"] = version_valid_loss
	validation_psnrs[f"v_{version}"] = version_valid_psnr

train_losses["iters"] = train_iters
validation_losses["epochs"] = validation_epochs
validation_psnrs["epochs"] = validation_epochs


nrows = len(train_losses["iters"])

for row in range(nrows):
	row_data = [f"{train_losses['iters'][row]}"]

	for version in range(args.model_versions):
		v = f"v_{version}"
		row_data.append(train_losses[v][row])
	train_writer.writerow(row_data)

nrows = len(validation_losses["epochs"])
for row in range(nrows):
	row_data = [f"{validation_losses['epochs'][row]}"]

	for version in range(args.model_versions):
		v = f"v_{version}"
		row_data.append(validation_losses[v][row])
	validation_writer.writerow(row_data)

for row in range(nrows):
	row_data = [f"{validation_psnrs['epochs'][row]}"]

	for version in range(args.model_versions):
		v = f"v_{version}"
		row_data.append(validation_psnrs[v][row])
	validation_psnr_writer.writerow(row_data)

