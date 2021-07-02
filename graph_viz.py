from graphviz import Digraph
from demosaic_ast import *
import argparse


def vis_ast(root, ids2nodes, name):
	graph = Digraph(name=name, format='png')
	print(root.dump())
	vis_ast_helper(root, ids2nodes, graph, 0)
	return graph

def vis_ast_helper(root, ids2nodes, graph, node_id, seen=None):
	if seen is None:
		seen = {}

	if id(root) in seen:
		return node_id
	else:
		root_id = node_id
		seen[id(root)] = root_id
		if issubclass(type(root), Linear) or issubclass(type(root), UnopIIdiv) or issubclass(type(root), Upsample) or issubclass(type(root), Downsample):
			in_c = root.in_c
			out_c = root.out_c
			if id(root) in ids2nodes:
				#node_label = f"{root.name}_{ids2nodes[id(root)]} {in_c} {out_c}"
				node_label = f"{root.name} ({in_c} {out_c})"
			else:
				node_label = f"{root.name} ({in_c} {out_c})"
			if hasattr(root, "groups"):
				node_label += f" g{root.groups}"
			if hasattr(root, 'factor'):
				node_label += f" s{root.factor}"
		else:
			node_label = root.name
		graph.node(str(node_id), node_label)
		node_id += 1


		if hasattr(root, 'node'):
			if id(root.node) in seen:
				child_id = seen[id(root.node)]
			else:
				child_id = node_id
				node_id = vis_ast_helper(root.node, ids2nodes, graph, node_id, seen)
			graph.edges([(str(child_id), str(root_id))])

		if root.num_children == 3:
			if id(root.child1) in seen:
				child1_id = seen[id(root.child1)]
			else:	
				child1_id = node_id	
				node_id = vis_ast_helper(root.child1, ids2nodes, graph, node_id, seen)
			if id(root.child2) in seen:
				child2_id = seen[id(root.child2)]
			else:
				child2_id = node_id
				node_id = vis_ast_helper(root.child2, ids2nodes, graph, node_id, seen)		
			if id(root.child3) in seen:
				child3_id = seen[id(root.child3)]
			else:
				child3_id = node_id
				node_id = vis_ast_helper(root.child3, ids2nodes, graph, node_id, seen)		

			graph.edges([(str(child1_id), str(root_id)), (str(child2_id), str(root_id)), (str(child3_id), str(root_id))])


		if root.num_children == 2:
			if id(root.lchild) in seen:
				lchild_id = seen[id(root.lchild)]
			else:	
				lchild_id = node_id	
				node_id = vis_ast_helper(root.lchild, ids2nodes, graph, node_id, seen)
			if id(root.rchild) in seen:
				rchild_id = seen[id(root.rchild)]
			else:
				rchild_id = node_id
				node_id = vis_ast_helper(root.rchild, ids2nodes, graph, node_id, seen)		

			graph.edges([(str(lchild_id), str(root_id)), (str(rchild_id), str(root_id))])


		elif root.num_children == 1:
			if id(root.child) in seen:
				child_id = seen[id(root.child)]
			else:
				child_id = node_id
				node_id = vis_ast_helper(root.child, ids2nodes, graph, node_id, seen)
			graph.edges([(str(child_id), str(root_id))])

	return node_id

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_ast", type=str)
	parser.add_argument("--outfile", type=str)
	args = parser.parse_args()

	model = load_ast(args.model_ast)
	nodes = model.preorder()
	ids2nodes = dict([(id(n), i) for i,n in enumerate(nodes)])

	graph = vis_ast(model, ids2nodes, args.outfile)
	model.compute_size(set(), count_all_inputs=True)
	print(model.size)
	graph.render()

