import os
import ast

class ModelDatabase():
	"""
	data storage format: array
	index i stores model id i's data
	The data is : (model_id
								model structural hash, 
								model occurence count, 
								best model version ID
								model losses for each version, 
								model compute cost, 
								model parent id)
	"""
	def __init__(self, database_dir, database_file=None):
		self.database = []
		self.database_dir = database_dir
		if not database_file is None:
			database_path = os.path.join(database_dir, database_file)
			self.load(database_path)

	def add(self, model_id, data):
		assert(len(database) == int(model_id)):
		self.database.append(data)

	def get_best_version_id(self, model_id):
		assert self.database[model_id][0] == model_id, \
				f"Model ID {self.database[model_id][0]} in unexpected location {model_id}"
		return self.database[model_id][2]

	def update_occurence_count(self, model_id, occurence_count):
		assert self.database[model_id][0] == model_id, \
				f"Model ID {self.database[model_id][0]} in unexpected location {model_id}"
		self.database[model_id][1] = occurence_count

	def save(self):
		database_file = 'snapshot-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
		database_path = os.path.join(self.database_dir, database_file)
		with open(database_path, "w+") as f:
			for item in self.database:
				f.write(f"{item[0]},{item[1]},{item[2]},{item[3]},{item[4]},{item[5]},{item[6]}")

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

				self.database.append((model_id, structural_hash, occurence_count,\
									 best_model_version, losses, compute_cost, parent_id))

