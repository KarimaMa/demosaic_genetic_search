import copy
from demosaic_ast import * 
from tree import *
import util
import os



def dump(node, indent="", printstr="", nodeid=None):
	if nodeid is None:
		nodeid = 0
	tab = "   "
	if not hasattr(node, 'in_c'):
		node.compute_input_output_channels()
	printstr += "\n {} {} {} {}".format(indent, node.name, node.in_c, node.out_c)
	if hasattr(node, "groups"):
		printstr += f" g{node.groups}"
	printstr += f"  [ID: {nodeid}] {id(node)}"

	nodeid += 1
	if node.num_children == 3:
		if node.child1 is None:
			return printstr

		printstr = dump(node.child1, indent+tab, printstr, nodeid)

		child1size = node.child1.compute_size(set(), count_input_exprs=True)
		nodeid += child1size
		printstr = dump(node.child2, indent+tab, printstr, nodeid)

		child2size = node.child2.compute_size(set(), count_input_exprs=True)
		nodeid += child2size
		printstr = dump(node.child3, indent+tab, printstr, nodeid)

	elif node.num_children == 2:
		if node.lchild is None:
			return printstr

		printstr = dump(node.lchild, indent+tab, printstr, nodeid)

		lchildsize = node.lchild.compute_size(set(), count_input_exprs=True)
		nodeid += lchildsize
		printstr = dump(node.rchild, indent+tab, printstr, nodeid)

	elif node.num_children == 1:
		if node.child is None:
			return printstr
		printstr = dump(node.child, indent+tab, printstr, nodeid)

	return printstr


def tree_structure(root):
	preorder_nodes = root.preorder()
	structure = ""
	for n in preorder_nodes:
		structure += f"-{n.name}-"
	return structure

"""
Given a node within a tree AST, and a dictionary of seen
subtrees, adds all unique subtrees with exactly N nodes
rooted at the given node 
"""
def enumerate_subtrees(node, count, N, seen):
	if count == N:	
		return []
	else:
		count += 1
		subtrees = []
		if isinstance(node, Binop):
			for i in range(count, N+1):
				if count == N:
					node_copy = copy.copy(node)
					if node.num_children == 2:
						node_copy.lchild = None
						node_copy.rchild = None
					elif node.num_children == 1:
						node_copy.child = None
					node_copy.num_children = 0
					return [node_copy]

				lchilds = enumerate_subtrees(node.lchild, i, N, seen) # 1, 2, 3
				rchilds = enumerate_subtrees(node.rchild, N-i+count, N, seen) # 3 - 1 + 1 = 3, 3 - 2 + 1 = 2
				for lchild in lchilds:
					for rchild in rchilds:
						node_copy = copy.copy(node)
						node_copy.parent = None

						node_copy.lchild = lchild
						node_copy.rchild = rchild
						subtrees += [node_copy]
						if count == 1: # the topmost function call
							seen[tree_structure(node_copy)] = node_copy

		elif isinstance(node, Unop):
			if count == N:
				node_copy = copy.copy(node)
				if node.num_children == 2:
					node_copy.lchild = None
					node_copy.rchild = None
				elif node.num_children == 1:
					node_copy.child = None
				node_copy.num_children = 0
				return [node_copy]

			childs = enumerate_subtrees(node.child, count, N, seen)
			for child in childs:
				node_copy = copy.copy(node)
				node_copy.parent = None
				node_copy.child = child
				subtrees += [node_copy]

				if count == 1: # the topmost function call
					seen[tree_structure(node_copy)] = node_copy
		else: # is input node
			node_copy = copy.copy(node)
			subtrees += [node_copy]

		return subtrees


def get_possible_op_pairs(ops):
	op_pairs = set()
	for op1 in ops:
		for op2 in ops:
			if op2 == op1 and not isinstance(op1, Binop):
				continue
			op_pairs.add((op1,op2))
			op_pairs.add((op2,op1))

	return op_pairs

def missing_neighbor_pairs(tree, missing_op_pairs):
	node_op = type(tree)

	if tree.parent:
		if type(tree.parent) is tuple:
			for p in tree.parent:
				parent_op = type(p)
			
				if (parent_op, node_op) in missing_op_pairs:
					missing_op_pairs.remove((parent_op, node_op))
		else:
			parent_op = type(tree.parent)
			if (parent_op, node_op) in missing_op_pairs:
				missing_op_pairs.remove((parent_op, node_op))			
	
	if tree.num_children == 3:
		children = [tree.child1, tree.child2, tree.child3]
	elif tree.num_children == 2:
		children = [tree.lchild, tree.rchild]
	elif tree.num_children == 1:
		children = [tree.child]
	else:
		children = []
	
	children_ops = [type(c) for c in children]

	for child_op in children_ops:
		if (node_op, child_op) in missing_op_pairs:
			missing_op_pairs.remove((node_op, child_op))

	for child in children:
		missing_neighbor_pairs(child, missing_op_pairs)


def find_missing_ops(tree, missing_ops):
	node_op = type(tree)
	if node_op in missing_ops:
		missing_ops.remove(node_op)

	if tree.num_children == 3:
		children = [tree.child1, tree.child2, tree.child3]
	elif tree.num_children == 2:
		children = [tree.lchild, tree.rchild]
	elif tree.num_children == 1:
		children = [tree.child]
	else:
		children = []
	
	for c in children:
		find_missing_ops(c, missing_ops)


def get_all_neighbors(root, parents, children, seen=None):
	if seen is None:
		seen = set()
	preorder_nodes = root.preorder()
	for n in preorder_nodes:
		if id(n) in seen:
			continue
		else:
			seen.add(id(n))
		node_type = type(n)
		if not node_type in parents:
			parents[node_type] = set()

		if root.parent:
			if type(root.parent) is tuple:
				for p in root.parent:
					parent_op = type(p)
					parents[node_type].add(parent_op)				
			else:
				parent_op = type(root.parent)
				parents[node_type].add(parent_op)
		
		if root.num_children == 3:
			node_children = [root.child1, root.child2, root.child3]
		elif root.num_children == 2:
			node_children = [root.lchild, root.rchild]
		elif root.num_children == 1:
			node_children = [root.child]
		else:
			node_children = []
		
		children_ops = [type(c) for c in node_children]
		
		if not node_type in children:
			children[node_type] = set()

		for child_op in children_ops:
			children[node_type].add(child_op)

		for c in node_children:
			get_all_neighbors(c, parents, children, seen)

	return parents, children



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str)

args = parser.parse_args()


missing_pairs = get_possible_op_pairs(all_ops)
print(f"number of missing pairs initially: {len(missing_pairs)}")
for model_id in os.listdir(args.model_dir):
	model_manager = util.ModelManager(args.model_dir, 0)
	try:	
		model = model_manager.load_model_ast(model_id)
	except:
		continue
	missing_neighbor_pairs(model, missing_pairs)

print(f"number of missing pairs after: {len(missing_pairs)}")
for mp in missing_pairs:
	print(mp)

missing_ops = all_ops

print(f"number of missing ops initially: {len(missing_ops)}")
for model_id in os.listdir(args.model_dir):
	model_manager = util.ModelManager(args.model_dir, 0)
	try:	
		model = model_manager.load_model_ast(model_id)
	except:
		continue
	find_missing_ops(model, missing_ops)

print(f"number of missing ops after: {len(missing_ops)}")
for mo in missing_ops:
	print(mo)

print(f"---- op neighbors ----")
parents = {}
children = {}
for model_id in os.listdir(args.model_dir):
	model_manager = util.ModelManager(args.model_dir, 0)
	try:	
		model = model_manager.load_model_ast(model_id)
	except:
		continue

	get_all_neighbors(model, parents, children)

for node in parents:
	print(f"node {node} has parents:")
	for p in parents[node]:
		print(p)
	print(f"node {node} has children:")
	for c in children[node]:
		print(c)


for tree_size in [4,5,6]:
	print(f"--- enumerating different subtrees of size {tree_size}")
	seen_subtrees = {}
	for model_id in os.listdir(args.model_dir):
		model_manager = util.ModelManager(args.model_dir, 0)
		try:	
			model = model_manager.load_model_ast(model_id)
		except:
			continue

		preorder_nodes = model.preorder()
		for node in preorder_nodes:
			enumerate_subtrees(node, 0, tree_size, seen_subtrees)
	filtered_subtrees = {}			
	for s, subtree in seen_subtrees.items():
		subtree_preorder = subtree.preorder()
		if len(subtree_preorder) == tree_size:
			filtered_subtrees[s] = subtree

	print(f"number of different subtrees of size {tree_size}: {len(filtered_subtrees)}")



