import csv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--subsets", type=int)
parser.add_argument("--model_versions", type=int)

args = parser.parse_args()

for subset in range(args.subsets):
	model_dir = os.path.join(args.data_dir, f"subset_{subset}")
	summary_train_csv = os.path.join(model_dir, f"all_versions_train.csv")
	summary_train_f = open(summary_train_csv, "w", newline="\n")
	summary_train_writer = csv.writer(summary_train_f, delimiter=",")

	summary_validation_csv = os.path.join(model_dir, f"all_versions_validation.csv")
	summary_validation_f = open(summary_validation_csv, "w", newline="\n")
	summary_validation_writer = csv.writer(summary_validation_f, delimiter=",")

	for version in range(args.model_versions):

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

				print(f"train iter {train_iter} loss {loss}")
				
				train_writer.writerow([train_iter, loss])  
				summary_train_writer.writerow([version, train_iter, loss])

		with open(validation_log, "r") as f:
			for l in f:
				print(l)

				if not "epoch" in l:
					continue
				epoch = int(l.split(' ')[5].strip())
				loss = float(l.split(' ')[6].strip())
				print(f"epoch {epoch} loss {loss}")

				validation_writer.writerow([epoch, loss])
				summary_validation_writer.writerow([version, epoch, loss])

