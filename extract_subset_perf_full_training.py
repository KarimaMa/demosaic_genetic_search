import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--eval_csv", type=str)
parser.add_argument("--input_data", type=str)
args = parser.parse_args()


eval_f = open(args.eval_csv, 'w', newline='\n')
eval_writer = csv.writer(eval_f, delimiter=',')

with open(args.input_data, "r") as f:
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
