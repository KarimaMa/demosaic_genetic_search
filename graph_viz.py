from graphviz import Digraph


def vis_ast(root, name):
	graph = Digraph(name=name, format='png')
	vis_ast_helper(root, graph, 0)
	return graph

def vis_ast_helper(root, graph, node_id, seen=None):
	if seen is None:
		seen = {}

	if id(root) in seen:
		return node_id
	else:
		root_id = node_id
		seen[id(root)] = root_id
		graph.node(str(node_id), root.name)
		node_id += 1

		if root.num_children == 2:
			if id(root.lchild) in seen:
				lchild_id = seen[id(root.lchild)]
			else:	
				lchild_id = node_id	
				node_id = vis_ast_helper(root.lchild, graph, node_id, seen)
			if id(root.rchild) in seen:
				rchild_id = seen[id(root.rchild)]
			else:
				rchild_id = node_id
				node_id = vis_ast_helper(root.rchild, graph, node_id, seen)		

			#graph.edges([(str(root_id), str(lchild_id)), (str(root_id), str(rchild_id))])
			graph.edges([(str(lchild_id), str(root_id)), (str(rchild_id), str(root_id))])


		elif root.num_children == 1:
			if id(root.child) in seen:
				child_id = seen[id(root.child)]
			else:
				child_id = node_id
				node_id = vis_ast_helper(root.child, graph, node_id, seen)
			#graph.edges([(str(root_id), str(child_id))])
			graph.edges([(str(child_id), str(root_id))])

	return node_id

if __name__ == "__main__":
	import model_lib
	model = model_lib.DecoupleExampleModel(2,10)

	graph = vis_ast(model, 'seed_model')
	model.compute_size(set(), count_all_inputs=True)
	print(model.size)
	graph.render()

