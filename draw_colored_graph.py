from tree_manipulation import get_children, get_parents
from type_check import compute_resolution
from graphviz import Digraph
from demosaic_ast import *
import argparse
from resolution_coloring import change_subgraph_resolution, flip_resolution, delete_resolution_subgraph, swap_resolution_op


# import pygraphviz as pgv

# """
# given a graph and a map from resolutions to colors, 
# draws the colored map
# """
# def draw_graph(graph, colormap):
# 	G = pgv.AGraph(directed=True)
# 	preorder_nodes = graph.preorder()
# 	nodemap = {}
# 	for i, n in enumerate(preorder_nodes):
# 		node_name = f'{n.name}_id{id(n)}'
# 		print(f'adding {node_name}')
# 		if not id(n) in nodemap:
# 			nodemap[id(n)] = node_name
# 			G.add_node(node_name, label=n.name, color=colormap[float(n.resolution)])
# 		for c in get_children(n):
# 			child_name = f'{c.name}_id{id(c)}'
# 			if not id(c) in nodemap:
# 				print(f'adding {id(c)}')
# 				nodemap[id(c)] = child_name
# 				G.add_node(child_name, label=c.name, color=colormap[float(c.resolution)])
# 			G.add_edge(child_name, node_name)
# 	G.layout()
# 	print(G.edges())
# 	print(len(G.nodes()))
# 	G.draw('foo.png')

def draw_graph(model, name):
	colormap = {float(1/6): 'red', float(1/3): 'blue', float(1/2): 'yellow', float(2/3): 'orange', float(1.0): 'green', float(2.0): 'purple'}
	nodes = model.preorder()
	ids2nodes = dict([(id(n), i) for i,n in enumerate(nodes)])
	graph = Digraph(name=name, format='png')
	print("drawing new model...")
	print(model.dump())
	draw_graph_helper(model, ids2nodes, graph, 0, colormap)
	graph.render(name)

def draw_graph_helper(root, ids2nodes, graph, node_id, colormap, seen=None):
	if seen is None:
		seen = {}

	if id(root) in seen:
		return node_id
	else:
		root_id = node_id
		seen[id(root)] = root_id
		root_label = root.name
		if isinstance(root, Downsample) or isinstance(root, Upsample):
			root_label += f'_f{root.factor}'
		graph.node(str(root_id), root_label, color=colormap[float(root.resolution)])
		
		node_id += 1

		children = get_children(root)

		for child in children:
			if id(child) in seen:
				child_id = seen[id(child)]
			else:
				child_id = node_id
				node_id = draw_graph_helper(child, ids2nodes, graph, node_id, colormap, seen)
			graph.edges([(str(child_id), str(root_id))])

	return node_id



if __name__ == "__main__":
	from xtrans_model_lib import *
	model = XGreenDemosaicknet3(3,8)
	compute_resolution(model)
	draw_graph(model, 'before')

	MAX_TRIES = 10
	change_subgraph_resolution(model, 2, MAX_TRIES)
	draw_graph(model, 'after-insert')

	# model = XGreenDemosaicknet3(3,8)
	try:
		flip_resolution(model)
		draw_graph(model, 'after-boundaryshift')
	except AssertionError as e:
		print(e)

	try:
		delete_resolution_subgraph(model)
		draw_graph(model, 'after-subgraphres-deletion')
	except AssertionError as e:
		print(e)

	try:
		swap_resolution_op(model)
		draw_graph(model, 'after-swapping-op')
	except AssertionError as e:
		print(e)
