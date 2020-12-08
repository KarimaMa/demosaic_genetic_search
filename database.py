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

	def add(self, key, data):
		self.table[key] = data

	def increment(self, key, fieldname):
		self.table[key][fieldname] += 1 

	def get(self, key, fieldname):
		return self.table[key][fieldname]

	def update(self, key, fieldname, value):
		if key in self.table: 
			self.table[key][fieldname] = value

	def save(self):
		database_f = open(self.database_path, "w", newline='\n')
		database_writer = csv.writer(database_f, delimiter=',')

		header = ["key"] + self.fields
		database_writer.writerow(header)
		for key, data in self.table.items():
			row_data = [key]
			row_data += [data[field_name] for field_name in self.fields]
			database_writer.writerow(row_data)

	def load(self, database_path):
		with open(database_path, "r+") as f:
			for i,l in enumerate(f):
				string_data = l.strip().split(",")
				if i == 0:
					continue
				key = int(string_data[0])
				string_data = string_data[1:]
				data = dict([(self.fields[j], self.field_types[j](string_data[j])) for j in range(len(self.fields))])
				self.table[key] = data




