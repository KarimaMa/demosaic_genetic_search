from orderedset import OrderedSet
import copy
import random 

"""
Let N be a node. 
Let C be one of its children.
Let X be another node.
Replace N's child C with X
"""
def replace_child(N, C, X):
	if N.num_children == 3:
		if id(C) == id(N.child1):
			N.child1 = X
		elif id(C) == id(N.child2):
			N.child2 = X
		else:
			N.child3 = X
	elif N.num_children == 2:
		if id(C) == id(N.lchild):
			N.lchild = X
		else:
			N.rchild = X
	else:
		N.child = X


"""
returns children of a node
"""
def get_children(node):
	if node.num_children == 3:
		return [node.child1, node.child2, node.child3]
	elif node.num_children == 2:
		return [node.lchild, node.rchild]
	elif node.num_children == 0:
		return [node.child]
	else:
		return []

"""
returns parents of node as a list
"""
def get_parents(node):
	if not type(node.parent) is tuple:
		parents = [node.parent]
	elif not node.partent is None:
		parents = list(node.parent)
	else:
		parents = []
	return parents


"""
returns the ids of nodes that would be dominated by node N if it was
inserted between the given child and parent pair: A, B
This function is used for finding nodes dominated by a proposed Upsample node prior
to its insertion. 
"""

def dominated_set(A):
	DominatedSet = OrderedSet()
	A_parents = get_parents(A)
	if len(A_parents) > 1: # a node with one child who has multiple parents does not dominate anything 
		return DominatedSet 

	DominatedSet.add(id(A))
	frontier = OrderedSet((A,))

	while len(frontier) > 0:
		new_frontier = OrderedSet()
		for node in frontier:
			parents = get_parents(node)
			parents_dominated = [id(p) in DominatedSet for p in parents]
			if all(parents_dominated):
				DominatedSet.add(id(node))
				children = get_children(node)
				for c in children:
					new_frontier.add(c)
		frontier = new_frontier

	return DominatedSet


"""
Given list of parents and a node, assigns the parents to the node
"""
def assign_parents(node, parents):
	if len(parents) == 1:
		node.parent = parents[0]
	else:
		node.parent = tuple(parents)

"""
Let S be a path presetned as a list of nodes in order from upstream to downstream computation.
Let A be the root node of S where there is a downstream path from A that goes through A->U->B
return all nodes on path S that are not dominated by the node U. 
"""
def get_undominated_nodes(S, A):
	dominated_by_U = dominated_set(A)
	undominated = OrderedSet([s for s in S if not id(s) in dominated_by_U])
	return undominated


"""
Let E be the node directly below which we are going to insert a downsample. 
A downsample will be inserted between E and each of its children if it has more than one child. 
Let A and B be the two nodes downstream of E between which we will insert a corresponding Upsample.
Let S be the list of nodes in increasing downstream order from E to A that defines the path of 
computation which we wish to evaluate at lower resolution. 

Return all the nodes that must be duplicated in order to decouple the computation of S from any other
paths in the graph that will not be dominated by the Upsample. 

VERY IMPORTANT: This algorithm assumes that all nodes dominated by the proposed Upsample MUST be 
downstream of the proposed Downsample.
"""
def get_nodes_to_decouple(S, E, A):
	# simply get the ndoes in S that are not dominated by the proposed upsample
	# and add any of their children that are not already in the undominated set
	undominated_in_S = get_undominated_nodes(S, A)
	nodes_to_decouple = undominated_in_S
	for s in undominated_in_S:
		children = get_children(s)
		for c in children:
			nodes_to_decouple.add(c)
	return nodes_to_decouple



"""
Let E be the node directly below which we are going to insert a downsample. 
A downsample will be inserted between E and each of its children if it has more than one child. 
Let A and B be the two nodes downstream of E between which we will insert a corresponding Upsample.
Let S be the list of nodes in increasing downstream order from E to A that defines the path of 
computation which we wish to evaluate at lower resolution. 

Decouple the computation of S from any other paths in the graph that will not be dominated by the Upsample. 

VERY IMPORTANT: This algorithm assumes that all nodes dominated by the proposed Upsample MUST be 
downstream of the proposed Downsample.
"""
def separate_path(S, E, A, B):
	nodes_to_decouple = get_nodes_to_decouple(S, E, A)
	# resort list in increasing upstream order, from proposed Upsample location towards the proposed Downsample location
	nodes_to_decouple = list(nodes_to_decouple).reverse()

	# ids of nodes dominated by proposed Upsample 
	dominated_node_ids = dominated_set(A)
	# keep a map of node IDs to their copies
	nodeID2copy = {}

	# ids of nodes in the path S
	S_ids = OrderedSet([id(s) for s in S])

	for n in nodes_to_decouple:
		nn = copy.copy(n)
		nodeID2copy[id(n)] = nn

		original_parents = get_parents(n)

		n_parent_dict = dict([(id(p), p) for p in original_parents])
		nn_parent_dict = dict([(id(p), p) for p in original_parents])

		# the copy nn keeps parents that are not dominated by the Upsample 
		# except for the edge  case where the copy is the copy of A which will detatch from B, since B belongs to the original A
		# the original n keeps parents that are dominated by the Upsample or are in the path S 
		# except in the edge case where the original is A which will keep B as a parent 
		for p in original_parents:
			if id(n) == id(A) and id(p) == id(B): # n keeps connection to B, nn removes connection to B
				del nn_parent_dict[id(p)]
			else:
				if id(p) in dominated_node_ids: 
					del nn_parent_dict[id(p)]
				elif not id(p) in S_ids: 
					del n_parent_dict[id(p)]

		n_parents = list(n_parent_dict.values())
		# separate computation by replacing parents of nn with their copied versions 
		# if the parent was copied and setting the child of the parent as nn instead of n
		nn_parents = []
		for parent_id in nn_parents:
			if parent_id in nodeID2copy:
				parent = nodeID2copy[parent_id]
			else:
				parent = nn_parent_dict[parent_id]
			nn_parents.append(parent)
			replace_child(parent, n, nn)

		assign_parents(n, n_parents)
		assign_parents(nn, nn_parents)


"""
let N be a Node in a graph, return the ids of all nodes downstream of N
"""
def get_downstream_nodes(N):
	parents = get_parents(N)
	downstream = OrderedSet([id(p) for p in parents])

	frontier = OrderedSet(parents)
	while len(frontier) > 0:
		new_frontier = OrderedSet()
		for n in frontier:
			parents = get_parents(p)
			for p in parents:
				new_frontier.add(p)
				downstream.add(id(p))
		frontier = new_frontier

	return downstream


"""
given a set of node ids and a list of nodes, returns the 
corresponding set of nodes
"""
def ids2nodes(node_ids, nodes):
	return OrderedSet([n for n in nodes if id(n) in node_ids])



"""
Given a graph, picks a random location insert a downsample and upsample
"""
def pick_down_up_insert_location(graph):
	preordered_nodes = graph.preorder()
	# insert downsample below E, between E and each of its children
	E = random.choice(preordered_nodes, 1)[0]
	downstream_ids_from_downsample = get_downstream_nodes(E).add(id(E))
	downstream_nodes_from_downsample = ids2nodes(downstream_ids_from_downsample, preordered_nodes)

	valid_upsample_children = []
	# filter out downstream locations that cannot be chosen as Upsample insertion location
	for node in downstream_nodes_from_downsample:
		dominated_ids = dominated_set(node)
		if dominated_ids.issubset(downstream_ids_from_downsample):
			valid_upsample_children.append(node)

	if len(valid_upsample_children) == 0:
		return None

	A = random.choice(valid_upsample_children, 1)[0]
	return E, A


"""
Let E be the upstream node and A be the dowstream node
Let C be a child of A
returns whether or not there is a path from A to E through C
"""
def path_exists(E, A, C):
	if id(C) == id(E):
		return True
	else:
		grand_children = get_children(C)
		return any([path_exists(C, E, gc) for gc in grand_children])
	

"""
returns list of paths between A and E
"""
def find_paths(E, A):
	paths = []
	# find all paths from A to E 
	children = get_children(A)
	for C in children:
		if id(C) == id(E):
			return [C]
		elif path_exists(E, A, C):
			# list of paths from E to C
			paths += find_paths(E, C)
	for path in paths:
		path.append(A)
			
	return paths


"""
inserts an up and downsample pair into the graph
returns None if fails to find working insert location for up and downsample
"""

def insert_down_upsample(graph, downsample_opclass, dowsample_kwargs, upsample_opclass, upsample_kwargs):
	locations = pick_down_up_insert_location(graph)
	if locations is None:
		return None

	E, A = locations
	paths = find_paths(E, A)
	# pick a path from E to A to do the lowres computation on
	S = random.choice(paths, 1)[0] # path begins with E and ends with A
	# pick a node B from parents of A to insert upsample between A and B
	parents_A = get_parents(A)
	B = random.choice(parents_A, 1)[0]

	separate_path(S, E, A, B)
	
	# insert downsample between E and each of its children 
	for child in get_children(E):
		downsample_op = downsample_opclass(child, **downsample_kwargs)
		downsample_op.parent = E
		replace_child(E, child, downsample_op)

	# insert upsample between A and B
	upsample_op = upsample_opclass(A, **upsample_kwargs)
	upsample_op.parent = B
	replace_child(B, A, upsample_op) 





