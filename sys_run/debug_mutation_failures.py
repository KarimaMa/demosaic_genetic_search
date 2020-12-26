import argparse
import logging
import sys
import importlib
sys.path.append(sys.path[0].split("/")[0])

import type_check
import demosaic_ast 
import mutate
import util
import model_lib
import cost


class Args():
	def __init__(self):
		self.min_subtree_size = 1
		self.max_subtree_size = 12
		self.max_footprint = 16
		self.subtree_selection_tries = 10
		self.default_channels = 16
		self.max_nodes = 35
		self.max_channels = 32
		self.green_model_asts = ['RETRAINED_GREEN_MODEL_SEARCH_12-22-COMBINED/models/3398/model_ast',
							     'RETRAINED_GREEN_MODEL_SEARCH_12-22-COMBINED/models/556/model_ast',
							     'RETRAINED_GREEN_MODEL_SEARCH_12-22-COMBINED/models/3318/model_ast',
							     'RETRAINED_GREEN_MODEL_SEARCH_12-22-COMBINED/models/833/model_ast']

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_dir", type=str)
	parser.add_argument("--model_id", type=str)
	parser.add_argument("--mutation_type", type=str)
	parser.add_argument("--insert_op", type=str)
	parser.add_argument("--insert_child_id", type=int)
	parser.add_argument("--delete_id", type=int)
	parser.add_argument("--chosen_conv_id", type=int)
	parser.add_argument("--decouple_node_id", type=int)

	args = parser.parse_args()

	log_format = '%(asctime)s %(levelname)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

	debug_logger = util.create_logger('debug_logger', logging.DEBUG, log_format, 'debug_log')
	mysql_logger = util.create_logger('mysql_logger', logging.INFO, log_format, 'myql_log')

	mutator_args = Args()
	mutator = mutate.Mutator(mutator_args, debug_logger, mysql_logger)
	mutator.current_mutation_info = mutate.MutationInfo()
	model_manager = util.ModelManager(args.model_dir, 0)
	tree = model_manager.load_model_ast(args.model_id)

	print("the original tree")
	print(tree.dump())
	old_tree = demosaic_ast.copy_subtree(tree)

	preorder_nodes = tree.preorder()
	print("nodes with multiple parents")
	for n in preorder_nodes:
		if type(n.parent) is tuple:
			print(n.dump())
	print("------------")

	print("partners in original tree:")
	for n in preorder_nodes:
		if hasattr(n, "partner_set"):
			print(f"node {n.name} partners:")
			for p, pid in n.partner_set:
				print(f"{p.name} {id(p)}")
	print("===========")

	if args.mutation_type == "insert":
		demosaic_ast = importlib.import_module("demosaic_ast")
		insert_op = getattr(demosaic_ast, args.insert_op)
		input_set = set(("Input(Bayer)",))
		new_tree = mutator.insert_mutation(tree, input_set, insert_above_node_id=args.insert_child_id, insert_op=insert_op)
	elif args.mutation_type == "delete":
		preorder_nodes = tree.preorder()
		print(f"attempting to delete\n{preorder_nodes[args.delete_id].dump()}")
		new_tree = mutator.delete_mutation(tree, preorder_nodes[args.delete_id])
	elif args.mutation_type == "decouple":
		print("decouple mutation")
		new_tree = mutator.decouple_mutation(tree, chosen_node_id=args.decouple_node_id)
	elif args.mutation_type == "channel_change":
		new_tree = mutator.channel_mutation(tree, chosen_conv_id=args.chosen_conv_id)
	elif args.mutation_type == "green_change":
		new_tree = mutator.green_model_change_mutation(tree)
	else:
		new_tree = mutator.group_mutation(tree)

	print("the new tree")
	print(new_tree.dump())
	print("the old tree")
	print(old_tree.dump())

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
			print("-----")
	print("===========")

	new_tree.compute_input_output_channels()
	type_check.check_channel_count(new_tree)
	accepted = mutator.accept_tree(new_tree)
	print(f"is tree accepted {accepted}")
	#type_check.check_linear_types(new_tree)
	type_check.assert_no_nonlinear_adjacency(new_tree)
	ev = cost.ModelEvaluator(None)
	cost_before = ev.compute_cost(new_tree)
	print(f"new tree cost before {cost_before}")
	new_tree.save_ast("test-ast")
	reloaded = demosaic_ast.load_ast("test-ast")
	print("reloaded new tree")
	print(reloaded.dump())

	cost_after = ev.compute_cost(reloaded)
	print(f"new tree cost after {cost_after}")

	preorder_nodes = reloaded.preorder()
	print("In reloaded tree nodes with multiple parents")
	for n in preorder_nodes:
		if type(n.parent) is tuple:
			print(n.dump())
	print("------------")

	print("partners in reloaded tree:")
	for n in preorder_nodes:
		if hasattr(n, "partner_set"):
			print(f"node {n.name} with id {id(n)} partners:")
			for p, pid in n.partner_set:
				print(f"{p.name} {id(p)}")
			print("-----")
	print("===========")
