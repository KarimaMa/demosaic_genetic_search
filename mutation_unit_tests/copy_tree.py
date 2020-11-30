import argparse
import logging
import sys
sys.path.append(sys.path[0].split("/")[0])

import demosaic_ast 
import mutate
import util
import model_lib


if __name__ == "__main__":
	log_format = '%(asctime)s %(levelname)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

	debug_logger = util.create_logger('debug_logger', logging.DEBUG, log_format, 'debug_log')
	mysql_logger = util.create_logger('mysql_logger', logging.INFO, log_format, 'myql_log')

	mutator = mutate.Mutator(None, debug_logger, mysql_logger)
	util.create_dir('junk_model')
	util.create_dir("junk_model/model")
	model = model_lib.MultiresQuadGreenModel(2,10)
	torch_model = model.ast_to_model().cuda()

	preorder_nodes = model.preorder()
	
	print("In original tree nodes with multiple parents")
	for n in preorder_nodes:
		if type(n.parent) is tuple:
			print(f"node {id(n)} parent {n.parent}{n.dump()}")
	print("------------")

	print("partners in new tree:")
	for n in preorder_nodes:
		if hasattr(n, "partner_set"):
			print(f"node {n.name} with id {id(n)} partners:")
			for p, pid in n.partner_set:
				print(f"{p.name} {id(p)}")
			print("-------")
	print("========")

	model_copy = demosaic_ast.copy_subtree(model)
	preorder_nodes = model_copy.preorder()
	
	print("In copy tree nodes with multiple parents")
	for n in preorder_nodes:
		if type(n.parent) is tuple:
			print(f"node {id(n)} parent {n.parent}{n.dump()}")
	print("------------")

	print("partners in copy tree:")
	for n in preorder_nodes:
		if hasattr(n, "partner_set"):
			print(f"node {n.name} with id {id(n)} partners:")
			for p, pid in n.partner_set:
				print(f"{p.name} {id(p)}")
			print("-------")
	print("========")

















