import csv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
args = parser.parse_args()

eval_log = os.path.join(args.data_dir, "eval_results")
eval_csv = os.path.join(args.data_dir, "eval_results.csv")

eval_f = open(eval_csv, 'w', newline='\n')
eval_writer = csv.writer(eval_f, delimiter=',')

with open(eval_log, "r") as f:
	for l in f:
		subset_id = l.split(" ")[1].strip()

		print(l)
		print(f"subset_id {subset_id}")

		train_data = l.split("training losses:")[1].strip()
		training_losses = [float(d.strip()) for d in train_data.split(',')]
		print("training losses")
		print(training_losses)
		print("-------------")
		eval_writer.writerow([subset_id] + training_losses)  
