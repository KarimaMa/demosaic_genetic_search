import csv

valid_f = open('subset_full_validation_performance', 'w', newline='\n')
valid_writer = csv.writer(valid_f, delimiter=',')

train_subset_f = open('subset_train_subset_performance', 'w', newline='\n')
train_writer = csv.writer(train_subset_f, delimiter=',')

with open("RAND_DATA_SUBSET_SELECTION/training_results", "r") as f:
  for i, l in enumerate(f):
    data = l.split("training losses:")[1].strip()
    train_data = data.split(" ")[0]
    validation_data = data.split(" ")[1]
    training_losses = [float(d.strip()) for d in train_data.split(',')]
    validation_losses = [float(d.strip()) for d in validation_data.split(',')]
    
    train_writer.writerow([i] + training_losses)  
    valid_writer.writerow([i] + validation_losses)

