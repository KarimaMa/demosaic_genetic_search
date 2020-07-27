import os
import ast
import time
import csv

class ModelDatabase():
	"""
	data storage format: array
	index i stores model id i's data
	The data is : (
							0	model_id
							1	model structural hash, 
							2	model occurence count, 
							3	best model version ID
							4	model losses for each version, 
							5	model compute cost, 
							6	model parent id,
							7	killed_children, // how many times this model could not be mutated to a new model
							8	failed_mutations, // how many mutations failed to get to this model
							9	prune_rejections, // how many mutations were pruned to get to this model
							10 structural_rejections, // how many mutations were rejected structurally to get to this model
							11 seen_rejections) // how many mutations were rejected as repeats to get to this model
	"""
	def __init__(self, database_dir, database_file=None):
		self.database = {}
		self.database_dir = database_dir
		if not database_file is None:
			database_path = os.path.join(database_dir, database_file)
			self.load(database_path)

	def add(self, model_id, data):
		self.database[model_id] = data

	def increment_killed_mutations(self, model_id):
		self.database[model_id][7] += 1 

	def get_best_version_id(self, model_id):
		return self.database[model_id][3]

	def update_occurence_count(self, model_id, occurence_count):
		if model_id in self.database: 
			self.database[model_id][2] = occurence_count

	def save(self):
		database_file = 'snapshot-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
		database_path = os.path.join(self.database_dir, database_file)

		database_f = open(database_path, "w", newline='\n')
		database_writer = csv.writer(database_f, delimiter=',')
		database_writer.writerow(["model_id",
								"structural_hash",
								"model_occurences",
								"best_version_id",
								"version_0_loss",
								"version_1_loss",
								"version_2_loss",
								"compute_cost",
								"parent_id",
								"killed_children",
								"failed_mutations",
								"prune_rejections",
								"structural_rejections",
								"seen_rejections"])

		for model_id, data in self.database.items():
			model_accuracies = data[4]
			row = data[0:4]
			for acc in model_accuracies:
				row += [acc]
			row += data[5:]
			database_writer.writerow(row)

		# with open(database_path, "w+") as f:
		# 	for model_id, data in self.database.items():
		# 		f.write(f"{data[0]},{data[1]},{data[2]}"+\
		# 				f"{data[3]},{data[4]},{data[5]}"+\
		# 				f"{data[6]},{data[7]},{data[8]}"+\
		# 				f"{data[9]},{data[10]},{data[11]}\n")

	def load(self, database_path):
		with open(database_path, "r+") as f:
			for l in f:
				data = l.strip().split(",")
				model_id = int(data[0])
				structural_hash = int(data[1])
				occurence_count = int(data[2])
				best_model_version = int(data[3])
				losses = ast.literal_eval(data[4])
				compute_cost = float(data[5])
				parent_id = int(data[6])

				self.database[model_id] = [model_id, structural_hash, occurence_count,\
									 best_model_version, losses, compute_cost, parent_id]

