import csv
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str)
parser.add_argument("--model_versions", type=int)
args = parser.parse_args()

for version in range(args.model_versions):
	train_log = os.path.join(args.data_dir, f"v{version}_train_log")
	train_csv = os.path.join(args.data_dir, f"v{version}_train.csv")
	train_f = open(train_csv, 'w', newline='\n')
	train_writer = csv.writer(train_f, delimiter=',')

	validation_log = os.path.join(args.data_dir, f"v{version}_validation_log")
	validation_csv = os.path.join(args.data_dir, f"v{version}_validation.csv")
	validation_f = open(validation_csv, 'w', newline='\n')
	validation_writer = csv.writer(validation_f, delimiter=',')

	with open(train_log, "r") as f:
		for l in f:
			train_iter = int(l.split(' ')[4].strip())
			loss = float(l.split(' ')[5].strip())

			print(l)
			print(f"train iter {train_iter} loss {loss}")
			
			train_writer.writerow([train_iter, loss])  

	with open(validation_log, "r") as f:
		for l in f:
			if not "epoch" in l:
				continue
			epoch = int(l.split(' ')[5].strip())
			loss = float(l.split(' ')[-1].strip())
			print(l)
			print(f"epoch {epoch} loss {loss}")

			validation_writer.writerow([epoch, loss])





