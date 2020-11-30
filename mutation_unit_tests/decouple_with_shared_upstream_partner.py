import argparse
import logging
import sys
sys.path.append(sys.path[0].split("/")[0])

from demosaic_ast import *
import mutate
import util
import model_lib


def build_model1():
	bayer = Input(4, "Bayer") # quad Bayer
	downsampled_bayer = Downsample(bayer)
	shared_conv = Conv2D(downsampled_bayer, 2, kwidth=3)
	softmax = Softmax(shared_conv)
	mul = Mul(softmax, shared_conv)
	conv2 = Conv2D(shared_conv, 2, kwidth=3)
	shared_upsample = Upsample(mul)
	upsample2 = Upsample(conv2)
	green_rb = Add(shared_upsample, upsample2)
	
	green_rb.compute_input_output_channels()

	downsampled_bayer.partner_set = set( [(shared_upsample, id(shared_upsample)), (upsample2, id(upsample2))] )
	shared_upsample.partner_set = set( [(downsampled_bayer, id(downsampled_bayer))] )
	upsample2.partner_set = set( [(downsampled_bayer, id(downsampled_bayer))] )

	bayer = Input(4, "Bayer")
	green = GreenExtractor(bayer, green_rb)
	green.assign_parents()
	green.compute_input_output_channels()

	return green 

def build_model2():
	bayer = Input(4, "Bayer") # quad Bayer'
	shared_conv = Conv2D(bayer, 2, kwidth=3)
	downsampled = Downsample(shared_conv)
	conv2 = Conv2D(downsampled, 2, kwidth=3)
	mul = Mul(downsampled, conv2)
	upsampled = Upsample(mul)
	upsampled.compute_input_output_channels()

	downsampled.partner_set = set( [(upsampled, id(upsampled))] )
	upsampled.partner_set = set( [(downsampled, id(downsampled))] )
	
	bayer = Input(4, "Bayer")
	green = GreenExtractor(bayer, upsampled)
	green.assign_parents()
	green.compute_input_output_channels()

	return green


if __name__ == "__main__":
	log_format = '%(asctime)s %(levelname)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

	debug_logger = util.create_logger('debug_logger', logging.DEBUG, log_format, 'debug_log')
	mysql_logger = util.create_logger('mysql_logger', logging.INFO, log_format, 'myql_log')

	mutator = mutate.Mutator(None, debug_logger, mysql_logger)

	print(" ----------  TEST 1 ---------- ")

	m = build_model1()
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

	new_tree = mutator.decouple_mutation(m)

	preorder_nodes = new_tree.preorder()
	print("In new tree nodes with multiple parents")
	for n in preorder_nodes:
		if type(n.parent) is tuple:
			print(n.dump())
	print("------------")

	print("partners in new tree:")
	for n in preorder_nodes:
		if hasattr(n, "partner_set"):
			print(f"node {n.name} with id {id(n)} partners:")
			for p, pid in n.partner_set:
				print(f"{p.name} {id(p)}")
			print("-------")
	print("========")


	print(" ----------  TEST 2 ---------- ")
	m = build_model2()
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
	
	new_tree = mutator.decouple_mutation(m)

	preorder_nodes = new_tree.preorder()
	print("In new tree nodes with multiple parents")
	for n in preorder_nodes:
		if type(n.parent) is tuple:
			print(n.dump())
	print("------------")

	print("partners in new tree:")
	for n in preorder_nodes:
		if hasattr(n, "partner_set"):
			print(f"node {n.name} with id {id(n)} partners:")
			for p, pid in n.partner_set:
				print(f"{p.name} {id(p)}")
			print("-------")
	print("========")

	
	