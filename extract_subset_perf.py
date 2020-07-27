import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--validation_csv", type=str)
parser.add_argument("--training_csv", type=str)
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

valid_f = open(args.validation_csv, 'w', newline='\n')
valid_writer = csv.writer(valid_f, delimiter=',')

train_subset_f = open(args.training_csv, 'w', newline='\n')
train_writer = csv.writer(train_subset_f, delimiter=',')

with open(args.input_data, "r") as f:
	for l in f:
		subset_id = l.split(" ")[1].strip()
		print(l)
		print(f"subset_id {subset_id}")

		data = l.split("training losses:")[1].strip()
		train_data = data.split("validation losses:")[0]
		validation_data = data.split("validation losses:")[1]
		training_losses = [float(d.strip()) for d in train_data.split(',')]
		validation_losses = [float(d.strip()) for d in validation_data.split(',')]
		print("training losses")
		print(training_losses)
		print("validation_losses")
		print(validation_losses)
		print("-----------")
		train_writer.writerow([subset_id] + training_losses)  
		valid_writer.writerow([subset_id] + validation_losses)
