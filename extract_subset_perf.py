import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--validation_result_file", type=str)
parser.add_argument("--training_result_file", type=str)
parser.add_argument("--input_data_file", type=str)
args = parser.parse_args()

valid_f = open(args.validation_result_file, 'w', newline='\n')
valid_writer = csv.writer(valid_f, delimiter=',')

train_subset_f = open(args.training_result_file, 'w', newline='\n')
train_writer = csv.writer(train_subset_f, delimiter=',')

with open(args.input_data_file, "r") as f:
  for i, l in enumerate(f):
    data = l.split("training losses:")[1].strip()
    train_data = data.split("validation losses:")[0].strip()
    validation_data = data.split("validation losses:")[1].strip()
    training_losses = [float(d.strip()) for d in train_data.split(',')]
    validation_losses = [float(d.strip()) for d in validation_data.split(',')]
    
    train_writer.writerow([i] + training_losses)  
    valid_writer.writerow([i] + validation_losses)

