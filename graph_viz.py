import torch
from graphviz import Digraph


def vis_ast(root, name):
	graph = Digraph(name=name)
	vis_ast_helper(root, graph, 0)
	return graph

def vis_ast_helper(root, graph, node_id):
	graph.node(str(node_id), root.name)
	root_id = node_id

	node_id += 1
	if root.num_children == 2:
		lchild_id = node_id
		node_id = vis_ast_helper(root.lchild, graph, node_id)

		rchild_id = node_id
		node_id = vis_ast_helper(root.rchild, graph, node_id)		

		graph.edges([(str(root_id), str(lchild_id)), (str(root_id), str(rchild_id))])

	elif root.num_children == 1:
		child_id = node_id
		node_id = vis_ast_helper(root.child, graph, node_id)
		graph.edges([(str(root_id), str(child_id))])
	return node_id

