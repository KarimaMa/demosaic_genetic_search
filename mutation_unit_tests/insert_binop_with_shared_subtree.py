import argparse
import logging
import sys
sys.path.append(sys.path[0].split("/")[0])

from demosaic_ast import *
import mutate
import util


def build_model():
	bayer = Input(4, "Bayer") # quad Bayer
	downsample1 = Downsample(bayer)
	conv1 = Conv2D(downsample1, 2, kwidth=3)

	bayer = Input(4, "Bayer") # quad Bayer
	downsample2 = Downsample(bayer)
	conv2 = Conv2D(downsample2, 2, kwidth=3)

	add = Add(conv1, conv2)
	upsample = Upsample(add)

	upsample.compute_input_output_channels()

	downsample1.partner_set = set( [(upsample, id(upsample))] )
	downsample2.partner_set = set( [(upsample, id(upsample))] )
	upsample.partner_set = set( [(downsample1, id(downsample1)), (downsample2, id(downsample2))] )

	bayer = Input(4, "Bayer")
	green = GreenExtractor(bayer, upsample)
	green.assign_parents()
	green.compute_input_output_channels()

	return green 

class Args():
	def __init__(self):
		self.min_subtree_size = 1
		self.max_subtree_size = 12
		self.subtree_selection_tries = 10
		self.default_channels = 16

if __name__ == "__main__":
	log_format = '%(asctime)s %(levelname)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

	debug_logger = util.create_logger('debug_logger', logging.DEBUG, log_format, 'debug_log')
	mysql_logger = util.create_logger('mysql_logger', logging.INFO, log_format, 'myql_log')

	args = Args()

	mutator = mutate.Mutator(args, debug_logger, mysql_logger)

	print(" ----------  INSERT MUTATION ---------- ")

	m = build_model()

	structure = m.structure_to_array()
	print(f"the model\n{m.dump()}")

	preorder_nodes = m.preorder()
	print("nodes with multiple parents")
	for n in preorder_nodes:
		if type(n.parent) is tuple:
			print(n.dump())
	print("------------")

	print("partners in original tree:")
	for n in preorder_nodes:
		if hasattr(n, "partner_set"):
			print(f"node {n.name} with id {id(n)} partners:")
			for p, pid in n.partner_set:
				print(f"{p.name} {id(p)}")
	print("========")

	model_inputs = set(("Input(Bayer)",))
	new_tree = mutator.insert_mutation(m, model_inputs, insert_above_node_id=3, insert_op=Mul)

	preorder_nodes = new_tree.preorder()
	print("========")
	print("In new tree nodes with multiple parents")
	for n in preorder_nodes:
		if type(n.parent) is tuple:
			print(f"\n{id(n)} parents {n.parent} {n.dump()}")
	print("------------")

	print("partners in new tree:")
	for n in preorder_nodes:
		if hasattr(n, "partner_set"):
			print(f"node {n.name} with id {id(n)} partners:")
			for p, pid in n.partner_set:
				print(f"{p.name} {id(p)}")
			print("-------")
	print("========")

	print(new_tree.dump())


	print(" -------- DECOUPLE MUTATION -------- ")
	new_tree = mutator.decouple_mutation(new_tree)
	preorder_nodes = new_tree.preorder()
	print("========")
	print("In decoupled tree nodes with multiple parents")
	for n in preorder_nodes:
		if type(n.parent) is tuple:
			print(f"\n{id(n)} parents {n.parent} {n.dump()}")
	print("------------")

	print("partners in decoupled tree:")
	for n in preorder_nodes:
		if hasattr(n, "partner_set"):
			print(f"node {n.name} with id {id(n)} partners:")
			for p, pid in n.partner_set:
				print(f"{p.name} {id(p)}")
			print("-------")
	print("========")

	print(new_tree.dump())	





