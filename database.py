import os
import ast
import time
import csv

class Database():
	"""
	data storage format: array
	index i stores model id i's data
	The data is values of given fields
	"""
	def __init__(self, name, fields, field_types, database_dir, database_file=None):
		self.name = name
		self.fields = fields
		self.field_types = field_types
		self.table = {}
		self.database_dir = database_dir
		if database_file is None:
			self.database_file = '{}-snapshot-{}'.format(self.name, time.strftime("%Y%m%d-%H%M%S"))
			self.database_path = os.path.join(self.database_dir, self.database_file)
		else:
			database_path = os.path.join(database_dir, database_file)
			self.load(database_path)


	def add(self, model_id, data):
		self.table[model_id] = data

	def increment(self, model_id, fieldname):
		self.table[model_id][fieldname] += 1 

	def get(self, model_id, fieldname):
		return self.table[model_id][fieldname]

	def update(self, model_id, fieldname, value):
		if model_id in self.table: 
			self.table[model_id][fieldname] = value

	def save(self):
		database_f = open(self.database_path, "w", newline='\n')
		database_writer = csv.writer(database_f, delimiter=',')

		header = self.fields
		database_writer.writerow(header)
		for model_id, data in self.table.items():
			row_data = [data[field_name] for field_name in self.fields]
			database_writer.writerow(row_data)

	def load(self, database_path):
		with open(database_path, "r+") as f:
			for i,l in enumerate(f):
				string_data = l.strip().split(",")
				if i == 0:
					continue
				data = dict([(self.fields[i], self.field_types[i](string_data[i])) for i in range(len(self.fields))])
				model_id = data["model_id"]
				self.table[model_id] = data




