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
Let N be a node. 
Let P be one of its parents.
Let X be another node.
Replace N's parent P with X
"""
def replace_parent(N, P, X):
	if type(N.parent) is tuple:
		parents = list(node.parent)
		new_parents = []
		for p in parents:
			if id(p) == id(P):
				new_parents += [X]
			else:
				new_parents += [p]
		N.parent = tuple(new_parents)
	else:
		N.parent = X



"""
New_node is being inserted between parent and child
Assumes new_node already points to child but makes the other
following edge changes to complete the insertion operation:

parent.child = new_node
new_node.parent = parent
child.parent = new_node
"""
def insertion_edge_updates(parent, child, new_node):
  replace_child(parent, child, new_node)
  new_node.parent = parent
  replace_parent(child, parent, new_node)


"""
returns children of a node
"""
def get_children(node):
	if node.num_children == 3:
		return [node.child1, node.child2, node.child3]
	elif node.num_children == 2:
		return [node.lchild, node.rchild]
	elif node.num_children == 1:
		return [node.child]
	else:
		return []

"""
returns parents of node as a list
"""
def get_parents(node):
	if node.parent is None:
		parents = []
	elif type(node.parent) is tuple:
		parents = list(node.parent)
	else:
		parents = [node.parent]
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
Let S be the list of all nodes along any path from E to A sorted in increasing downstream order that defines the  
computational subgraph which we wish to evaluate at lower resolution. 

Return all the nodes that must be duplicated in order to decouple the computation of S from any other
paths in the graph that will not be dominated by the Upsample. 

VERY IMPORTANT: This algorithm requires that all children of nodes in S MUST be contained by the set S + children of E
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
Let S be the list of all nodes along any path from E to A sorted in downstream topo order from A to E
that defines the computational subgraph which we wish to evaluate at lower resolution. 

Decouple the computation of S from any other paths in the graph that will not be dominated by the Upsample. 

VERY IMPORTANT: This algorithm requires that all children of nodes in S MUST be contained by the set S + children of E
"""
def separate_path(S, E, A, B):
	nodes_to_decouple = get_nodes_to_decouple(S, E, A)
	nodes_to_decouple = list(nodes_to_decouple)

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
def get_downstream_node_ids(N):
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
returns if A is downstream of B
"""
def is_downstream(A, B):
	parents = get_parents(A)
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

	return id(B) in downstream


"""
topologically sorts nodes in S from Downstream to Upstream
"""
def toposort(S, E, A):
	children_of_E = get_children(E)
	done = OrderedSet(children_of_E)
	pending = OrderedSet()

	while len(done) < (len(S) + len(children_of_E)):
		for n in done:
			consumers = get_parents(n)
			for c in consumer:
				if not c in done:
					pending.add(c)
		for n in pending:
			producers = get_children(p)
			if all([p in done for p in producers]):
				done.add(p)
				pending.remove(p)

	# remove children of E from sorted list since we only want nodes in S
	for c in children_of_E:
		done.remove(c)
		
	sorted_S = list(done) # nodes are in order from upstream to downstream, reverse it
	sorted_S.reverse()

	return OrderedSet(sorted_S)

"""
given a set of node ids and a list of nodes, returns the 
corresponding set of nodes
"""
def ids2nodes(node_ids, nodes):
	return OrderedSet([n for n in nodes if id(n) in node_ids])


"""
Given a graph, picks a random location insert a downsample and upsample
Let E be the node below which the downsample will be inserted
Let A be the node above which the upsample will be inserted
Let S be the set of nodes that feed into A from E
returns E, A, and S
"""
def pick_down_up_insert_location(graph):
	preordered_nodes = graph.preorder()
	# insert downsample below E, between E and each of its children
	E = random.choice(preordered_nodes, 1)[0]

	downstream_ids_from_downsample = get_downstream_node_ids(E).add(id(E))
	downstream_nodes_from_downsample = ids2nodes(downstream_ids_from_downsample, preordered_nodes)

	children_of_E = get_children(E)

	# Upsample can be inserted above any node A that is downstream of E
	# if the set S of nodes that feed into A from E have children that are either contained in S 
	# or are the children of E so that no other downsamples need to be inserted 
	allowedA = []

	for A in downstream_nodes_from_downsample:
		paths = find_paths(E, A)

		S_ids = []
		for path in paths:
			S_ids += path

		S_ids = OrderedSet(S_ids)	
		S = ids2nodes(S_ids, preordered_nodes)

		allowed_child_ids = S_ids | OrderedSet([id(c) for c in children_of_E])

		contained = []
		for s in S:
			s_children = get_children(s)
			if all([id(c) in allowed_child_ids for c in s_children]):
				contained.append(True)
			else:
				contained.append(False)
				break
		if all(contained):
			allowedA.append((A, S))

	if len(allowedA) == 0:
		return None
	# randomly pick a location A
	A, S = random.choice(allowedA, 1)[0]
	return E, A, S



"""
returns list of paths between A and E 
path is represented as list of node ids
"""
def find_paths(E, A):
	paths = []
	# find all paths from A to E 
	children = get_children(A)
	for C in children:
		if id(C) == id(E):
			paths.append( [id(C)] ) 
		else:
			paths_through_C = find_paths(E, C) # list of paths
			if len(paths_through_C) != 0:
				paths += paths_through_C

	for path in paths:
		path.append( id(A) )
			
	return paths


"""
inserts an up and downsample pair into the graph
returns None if fails to find working insert location for up and downsample
"""

def insert_down_upsample(graph, downsample_opclass, dowsample_kwargs, upsample_opclass, upsample_kwargs):
	preordered_nodes = graph.preorder()

	locations = pick_down_up_insert_location(graph)
	if locations is None:
		return None

	E, A, S = locations

	# pick a node B from parents of A to insert upsample between A and B
	parents_A = get_parents(A)
	B = random.choice(parents_A, 1)[0]

	sortedS = toposort(S, E, A)
	S_ids = OrderedSet([id(s) for s in sortedS])
	separate_path(sortedS, E, A, B)
	
	# insert downsample between E and each of its children 
	# also insert downsample between children of E and any nodes in S that consume them
	children_of_E = get_children(E)
	for child in children_of_E:
		parents = get_parents(child)
		assert all([id(p) in S_ids for p in parents]) # children of downsample should only feed into nodes within S
		for p in parents:
			downsample_op = downsample_opclass(child, **downsample_kwargs)
			downsample_op.parent = p
			replace_child(p, child, downsample_op)
			replace_parent(child, p, downsample_op)


	# insert upsample between A and B
	upsample_op = upsample_opclass(A, **upsample_kwargs)
	upsample_op.parent = B
	replace_child(B, A, upsample_op) 





