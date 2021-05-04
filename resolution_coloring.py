from demosaic_ast import downsample_ops, upsample_ops, LearnedDownsample, Pack, Unpack, LearnedUpsample, BilinearUpsample, Downsample, Upsample, border_ops
from tree_manipulation import get_children, get_parents, insertion_edge_updates, replace_parent, replace_child
from orderedset import OrderedSet
import random
from type_check import compute_resolution


FULLRESWIDTH = 120

class ColoredGraph:
	def __init__(self, color):
		self.children = []
		self.parents = []
		self.color = color

	def is_upstream(self, other):
		if id(self) == id(other):
			return True
		parents = self.parents
		if any([p.is_upstream(other) for p in parents]):
			return True
		return False

"""
given a graph G, returns:
CG: a "colored" version of G called where upsamples and downsamples are removed 
	and all nodes are marked with their resolution.
CG2G a map from nodes in CG to G
"""
def color_graph(g_node, visited, colored_parent):
	cg2g = {}

	# skip over downsample upsample nodes
	if isinstance(g_node, Downsample) or isinstance(g_node, Upsample):
		colored_node, cg2g = color_graph(g_node.child, visited, colored_parent)
	else:
		if id(g_node) in visited: # visited this node before but not from this parent, update parents list
			colored_node = visited[id(g_node)]
			if colored_parent:
				colored_node.parents.append(colored_parent)
		else: # never visited this node before
			colored_node = ColoredGraph(g_node.resolution)
			if colored_parent:
				colored_node.parents.append(colored_parent)

			cg2g[colored_node] = g_node
			visited[id(g_node)] = colored_node

			children = get_children(g_node)

			for c in children:
				colored_child, child_cg2g = color_graph(c, visited, colored_node)
				colored_node.children.append(colored_child)
				cg2g.update(child_cg2g)

	return colored_node, cg2g


"""
Given nodes upstream node U and downstream node D in the graph, 
returns all nodes along any paths from E to S
"""
def find_subgraph(U, D):
	nodes = OrderedSet()
	nodes.add(U)
	if id(U) == id(D):
		return nodes
	else:
		parents = get_parents(U)
		for parent in parents: 
			nodes_thru_parent = find_subgraph(parent, D)
			if len(nodes_thru_parent) > 0:
				nodes = nodes | nodes_thru_parent
	return nodes


"""
Given a set of nodes S that defines a subgraph, returns all
incoming and outgoing edges from the subgraph
"""
def get_incoming_outgoing_edges(S):
	incoming = OrderedSet()
	outgoing = OrderedSet()

	for node in S:
		parents = get_parents(node)
		for p in parents:
			if not p in S:
				outgoing.add((node, p))

		children = get_children(node)
		for c in children:
			if not c in S:
				incoming.add((c, node))

	return incoming, outgoing


"""
compute pixel width of the output of the given node 
given the pixel width of an output with full resolution 
i.e. resolution = 1
"""
def get_pixel_width(node, fullres_width):
	return node.resolution * fullres_width

"""
given a chosen downsample / upsample type, the scale factor, output resolution,
and the input child, generates the downsample / sample op
"""
def build_updown_op(OpType, factor, resolution, child):
	kwargs = {"factor": factor, "resolution": resolution}
	if OpType is LearnedDownsample or OpType is LearnedUpsample:
		kwargs["out_c"] = child.out_c 
	return OpType(child, **kwargs)


def insert_downsample(dest, source, factor, DownsampleTypes=None):
	if DownsampleTypes is None:
		possible_dowsample_types = downsample_ops
		pixel_width = get_pixel_width(source, FULLRESWIDTH)
		if pixel_width % factor != 0:
			possible_dowsample_types = possible_dowsample_types - OrderedSet((Pack,))
		DownsampleType = random.choice(possible_dowsample_types)
	else:
		DownsampleType = random.choice(DownsampleTypes)
	new_resolution = source.resolution / factor
	new_downsample = build_updown_op(DownsampleType, factor, new_resolution, source)
	insertion_edge_updates(dest, source, new_downsample)


def insert_upsample(dest, source, factor, UpsampleTypes=None):
	if UpsampleTypes is None:
		possible_upsample_types = upsample_ops
		channel_count = source.out_c
		if channel_count % (factor**2)!= 0:
			possible_upsample_types = possible_upsample_types - OrderedSet((Unpack,))
		UpsampleType = random.choice(possible_upsample_types)
	else:
		UpsampleType = random.choice(UpsampleTypes)
	new_resolution = source.resolution * factor
	new_upsample = build_updown_op(UpsampleType, factor, new_resolution, source)
	insertion_edge_updates(dest, source, new_upsample)


"""
deletes any up or downsample between dest and source
"""
def remove_updown_sample_between(dest, source):
	for n in get_children(dest):
		if isinstance(n, Upsample) or isinstance(n, Downsample):
			for nn in get_children(n):
				if id(nn) == id(source):
					replace_parent(nn, n, dest)
					replace_child(dest, n, source)


def touches_resolution_op(node):
	if any([isinstance(c, Downsample) or isinstance(c, Upsample) for c in get_children(node)]):
		return True
	if any([isinstance(p, Downsample) or isinstance(p, Upsample) for p in get_parents(node)]):
		return True


def touches_one_resolution_op(node):
	children_touches = sum([isinstance(c, Downsample) or isinstance(c, Upsample) for c in get_children(node)])
	parent_touches = sum([isinstance(p, Downsample) or isinstance(p, Upsample) for p in get_parents(node)])
	return (children_touches + parent_touches == 1)
		 

"""
Choose nodes A and downstream node B in the graph, all paths from A to B 
define a computational subgraph S. Change the resolution of S by inserting
up/downsamples a the boundary of S.

factor is the downsampling factor

Assumptions:
all nodes in the subgraph defined by A and B have the same resolution 
prior to insertion and our algorithm guarantees that they will all 
have the same resolution after insertion as well.
"""
def change_subgraph_resolution(graph, factor, MAX_TRIES):
	print("CHANGE SUBGRAPH RESOLUTION")
	g2cg = {}
	CG, cg2g = color_graph(graph, g2cg, None)
	# pick a random subgraph by selecting nodes A and B from the same connected component
	colored_nodes = list(cg2g.keys())

	tries = 0
	while tries < MAX_TRIES:
		node1 = random.choice(colored_nodes)
		# don't insert downsample below any nodes that are already adjacent to a resolution op
		# we want to avoid having resolution ops next to each other
		if touches_resolution_op(cg2g[node1]):
			continue
		
		connected_component = get_component(node1, cg2g, g2cg)

		# connected_component, cc_boundary_nodes = get_connected_component(node1)
		node2 = random.choice(connected_component.nodes)
		
		if touches_resolution_op(cg2g[node2]):
			continue

		if len(node1.parents) == 0 or len(node2.parents) == 0:
			continue
		if len(node1.children) == 0 or len(node2.children) == 0:
			continue

		if node1.is_upstream(node2):
			A = cg2g[node1]
			B = cg2g[node2]
		else:
			assert node2.is_upstream(node1)
			A = cg2g[node2]
			B = cg2g[node1]

		break

	print(f"node A: {A.dump()}")
	print(f"node B: {B.dump()}")
	S = find_subgraph(A, B)
	print(f"nodes in S:")
	for s in S:
		print(f"{s.dump()}")

	current_resolution = A.resolution
	new_resolution = A.resolution  / factor

	# find all incoming and outgoing edges from the subgraph
	incoming, outgoing = get_incoming_outgoing_edges(S)

	# dowsamples must be inserted along all edges in incoming
	# upsamples must be insserted along all edges in outgoing 
	for edge in incoming:
		child, parent = edge
		insert_downsample(parent, child, factor)

	for edge in outgoing:
		child, parent = edge
		insert_upsample(parent, child, factor)

	graph.compute_input_output_channels()
	compute_resolution(graph)


"""
Given a source and destination node, either adds or removes up or downsamples 
between them make sure the graph obeys their assigned resolutions
"""
def fix_resolutions(dest, source, upsample_types=None, downsample_types=None):
	if dest.resolution != source.resolution:
		if dest.resolution < source.resolution:
			factor = source.resolution / dest.resolution
			ok = check_for_downsample(dest, source, factor)
			if not ok: # insert downsample between source and dest
				insert_downsample(dest, source, factor, downsample_types)
		else:
			factor = dest.resolution / source.resolution
			ok = check_for_upsample(dest, source, factor)
			if not ok:
				insert_upsample(dest, source, factor, upsample_types)

	if dest.resolution == source.resolution:
		# check there is no upsample or downsample between them 
		ok = check_same_resolution(source, dest)
		if not ok: # delete any up / downsamples between child and fliped node
			remove_updown_sample_between(dest, source)



"""
given a node and a grandparent node, checks that there is an upsample directly 
connecting these two nodes with the given factor
"""
def check_for_upsample(grandparent, node, factor):
	children = get_children(grandparent)
	for child in children:
		if id(child) == id(node): # node cannot be adjacent to grandparent
			return False
		if isinstance(child, Upsample) and child.factor == factor:
			for grandchild in get_children(child):
				if id(node) == id(grandchild):
					return True
	return False

"""
given a node and a grandparent node, checks that there is a downsample directly 
connecting these two nodes with the given factor
"""
def check_for_downsample(grandparent, node, factor):
	children = get_children(grandparent)
	for child in children:
		if id(child) == id(node): # node cannot be adjacent to grandparent
			return False
		if isinstance(child, Downsample) and child.factor == factor:
			for grandchild in get_children(child):
				if id(node) == id(grandchild):
					return True
	return False


"""
given nodes source, dest in the graph, checks that there is no upsample or 
downsamnple between them. 
"""
def check_same_resolution(dest, source):
	children = get_children(dest)
	child_ids = [id(c) for c in children]
	if not id(source) in child_ids:
		return False

	for child in children:
		if isinstance(child, Upsample) or isinstance(child, Downsample):
			for grandchild in get_children(child):
				if id(grandchild) == id(source):
					return False
	return True



"""
let N be a boundary node in the colored graph, picks an adjacent 
color that is different from N's color
returns the adjacent node and color 
"""
def pick_adjacent_color(node):
	adjacent_colors = {} # dictionary mapping adjacent colors to their nodes

	for c in node.children:
		if c.color != node.color:
			if not c.color in adjacent_colors:
				adjacent_colors[c.color] = [c]
			else:
				adjacent_colors[c.color] += [c]
	for p in node.parents:
		if p.color != node.color:
			if not p.color in adjacent_colors:
				adjacent_colors[p.color] = [p]
			else:
				adjacent_colors[p.color] += [p]

	new_color = random.choice(list(adjacent_colors.keys()))
	return new_color, adjacent_colors[new_color]


"""
Given a colored graph CG, returns all nodes that are on color boundaries
"""
def find_boundary_nodes(node, visited):
	boundary_nodes = OrderedSet()
	if not node in visited:
		visited.add(node)
		for child in node.children:
			if child.color != node.color:
				boundary_nodes.add(node)
				boundary_nodes.add(child)
			boundary_nodes_from_child = find_boundary_nodes(child, visited)		
			boundary_nodes = boundary_nodes | boundary_nodes_from_child
	return boundary_nodes



"""
Upsample and Downsample nodes don't exist in the colored graph. We move nodes inside and outside
resolution boundaries by changing their color. Any nodes that have different colors and touch must 
have an up or downsample inserted between them if there isn't one there already 

Let G be a graph 
Let CG be its colored version
Select a random node N on the boundary of a resolution subgraph of CG
Change N's resolution to that of one of its neighbors with a different resolution
"""
def flip_resolution(G):
	print("CHANGING BOUNDARY")
	g2cg = {}
	CG, cg2g = color_graph(G, g2cg, None)

	boundary_nodes = find_boundary_nodes(CG, OrderedSet())
	# filter out any nodes that are border nodes - these cannot be flipped
	boundary_nodes = [n for n in boundary_nodes if not type(cg2g[n]) in border_ops]
	# don't flip nodes that touch more than one resolution ops because this would introduce adjacent resolution ops
	boundary_nodes = [n for n in boundary_nodes if touches_one_resolution_op(cg2g[n])]

	if len(boundary_nodes) == 0:
		assert False # no nodes to flip

	# randomly pick one to flip
	flip_node = random.choice(boundary_nodes)
	graph_node = cg2g[flip_node]
	# pick an adjacent different resolution and flip to that resolution
	new_resolution, adjacent_nodes = pick_adjacent_color(flip_node)
	print(f"flipping node")
	print(graph_node.dump())
	print(f"in tree\n{G.dump()}")
	print(f"to new resolution: {new_resolution}")
	graph_node.resolution = new_resolution

	# 1) if a node has a neighbor in CG with the same color but an up/downsample in between them in G, delete the up / downsample 
	# 2) if a node has a neighbor in CG with a different color but no up/downsample in between in G, insert the up/downsample needed	
	# only possibly affected nodes are the ones neighboring the node we just flipped so just check that node's parents and children
	children = flip_node.children
	parents = flip_node.parents

	for child in children:
		source = cg2g[child]
		dest = cg2g[flip_node]
		fix_resolutions(dest, source)

	for parent in parents:
		source = cg2g[flip_node]
		dest = cg2g[parent]
		fix_resolutions(dest, source)

	G.compute_input_output_channels()
	compute_resolution(G)


class Component:
	def __init__(self, nodes, boundary_nodes, subcomponent=None):
		self.nodes = nodes
		self.boundary_nodes = boundary_nodes
		self.subcomponent = subcomponent



def found_upstream_downsample(gnode):
	if isinstance(gnode, Downsample):
		return True

	return any([found_upstream_downsample(n) for n in get_children(gnode)])


"""
Given a node from the graph, finds the 
innermost upsample of the component it belongs to 


find furthest upstream upsample U if there is one
if there is a downsample upstream from U, return U
"""
def get_innermost_upsamples(gnode):
	innermost_upsamples = []
	if isinstance(gnode, Downsample):
		return innermost_upsamples
	else:
		children = get_children(gnode)
		for child in children:
			found = get_innermost_upsamples(child)
			innermost_upsamples += found
		
		if len(innermost_upsamples) == 0 and isinstance(gnode, Upsample):
			if found_upstream_downsample(gnode):
				return [gnode]

		return innermost_upsamples


def get_innermost_component(node, visited=None):
	if visited is None:
		visited = OrderedSet()

	component_nodes = OrderedSet((node,))
	boundary_nodes = OrderedSet()

	if node in visited:
		return component_nodes, boundary_nodes

	visited.add(node)

	for child in node.children:
		if child.color == node.color:
			nodes_from_child, boundary_nodes_from_child = get_innermost_component(child, visited)
			component_nodes |= nodes_from_child
			boundary_nodes |= boundary_nodes_from_child
		else:
			boundary_nodes.add(child)

	for parent in node.parents:
		if parent.color == node.color:
			nodes_from_parent, boundary_nodes_from_parent = get_innermost_component(parent, visited)
			component_nodes |= nodes_from_parent
			boundary_nodes |= boundary_nodes_from_parent
		else:
			boundary_nodes.add(parent)

	return component_nodes, boundary_nodes


"""
Returns 
1) nodes that are connected and in the same component as the given node
   that are not in the given enclosed subcomponent 
2) the boundary nodes of the component not including the boundary with the enclosed subcomponent
"""
def expand_component(node, subcomponent, visited=None):
	component_nodes = OrderedSet((node,))
	boundary_nodes = OrderedSet()
	
	if visited is None:
		visited = OrderedSet()

	if node in visited:
		return component_nodes, boundary_nodes

	visited.add(node)

	for child in node.children:
		if child.color == node.color:
			nodes_from_child, boundary_nodes_from_child = expand_component(child, subcomponent, visited)
			component_nodes |= nodes_from_child
			boundary_nodes |= boundary_nodes_from_child
		else:
			if not child in subcomponent.nodes:
				boundary_nodes.add(child)

	for parent in node.parents:
		if parent.color == node.color:
			nodes_from_parent, boundary_nodes_from_parent = expand_component(parent, subcomponent, visited)
			component_nodes |= nodes_from_parent
			boundary_nodes |= boundary_nodes_from_parent
		else:
			if not parent in subcomponent.nodes:
				boundary_nodes.add(parent)

	return component_nodes, boundary_nodes


"""
Given a node in a colored graph, returns its connected colored component 
"""
def get_component(cg_node, cg2g, g2cg):
	# expand from inside out
	innermost_upsamples = get_innermost_upsamples(cg2g[cg_node])
	if len(innermost_upsamples) > 0:
		innermost_upsample = innermost_upsamples[0]
		innermost_component_node = g2cg[id(innermost_upsample.child)]
		innermost_component_nodes, innermost_boundary = get_innermost_component(innermost_component_node)

		curr_nodes = innermost_component_nodes
		curr_boundary = innermost_boundary
		curr_subcomponent = Component(innermost_component_nodes, innermost_boundary)

		while True:
			new_nodes = OrderedSet()
			new_boundary = OrderedSet()
			for node in curr_boundary:
				component_nodes, boundary_nodes = expand_component(node, curr_subcomponent)
				new_nodes |= component_nodes
				new_boundary |= boundary_nodes

			curr_nodes |= new_nodes
			new_subcomponent = Component(curr_nodes, new_boundary, subcomponent=curr_subcomponent)

			curr_subcomponent = new_subcomponent
			curr_boundary = new_boundary

			if cg_node in curr_nodes:
				break
			
		return curr_subcomponent
	else: # there are no innermost upsamples, this is the innermost component
		component_nodes, boundary_nodes = get_innermost_component(cg_node)
		return Component(component_nodes, boundary_nodes)



def outof_component_source(gnode, g2cg, component_nodes):
	if isinstance(gnode, Upsample) or isinstance(gnode, Downsample):
		source = gnode.child
		if g2cg[id(source)] in component_nodes:
			return source
	return None



def into_component_dests(gnode, g2cg, component_nodes):
	if isinstance(gnode, Upsample) or isinstance(gnode, Downsample):
		dests = []
		for p in get_parents(gnode):
			if g2cg[id(p)] in component_nodes:
				dests += [p]
		if len(dests) > 0:
			return dests
	return None


"""
Given nodes A and B in the graph, all paths from B to A define 
a computational subgraph S. Change the resolution of S by inserting
up/downsamples a the boundary of S.

factor is the downsampling factor

Given a graph G, pick a resolution subgraph to delete such that all nodes 
in that subgraph have the same resolution as the nodes in the enclosing subgraph. 
"""
def delete_resolution_subgraph(G):
	print("inside delete subgraph")

	g2cg = {}
	CG, cg2g = color_graph(G, g2cg, None)

	# pick a random subgraph by selecting a random downsample node from the original graph
	# and finding the connected subcomponent defined by it 
	preorder_nodes = G.preorder()
	downsamples = [n for n in preorder_nodes if isinstance(n, Downsample)]

	if len(downsamples) == 0:
		assert False, "No downsamples in graph, cannot remove any resolution subgraph"

	chosen_downsample = random.choice(downsamples)
	new_resolution = chosen_downsample.child.resolution

	downsample_parents = get_parents(chosen_downsample)
	print(f"deleting subgraph bounded by {id(chosen_downsample)}")

	node_in_component = g2cg[id(downsample_parents[0])]

	component = get_component(node_in_component, cg2g, g2cg)

	# subgraph, subgraph_boundary_nodes = get_connected_component(node_in_component) 
	print(f"size of subgraph: {len(component.nodes)}")

	# find boundary nodes of the chosen subgraph and remove any adjacent upsamples or downsamples  
	# subgraph_boundary_nodes = [n for n in boundary_nodes if n in subgraph]
	print(f"removing subgraph with boundary nodes ")
	print([id(cg2g[n]) for n in component.boundary_nodes])
	outgoing_boundary_nodes = [n for n in component.boundary_nodes if any([c in component.nodes for c in n.children])]
	incoming_boundary_nodes = [n for n in component.boundary_nodes if any([p in component.nodes for p in n.parents])]

	print(f"incoming boundary nodes: {[id(cg2g[n]) for n in incoming_boundary_nodes]}")
	print(f"outgoing boundary nodes: {[id(cg2g[n]) for n in outgoing_boundary_nodes]}")

	for boundary_node in outgoing_boundary_nodes:
		graph_node = cg2g[boundary_node]
		children = get_children(graph_node)

		for c in children: 
			source = outof_component_source(c, g2cg, component.nodes)
			if source:			
				dest = graph_node
				if dest.resolution == new_resolution:
					print(f'removing resolution node between {id(dest)} and {id(source)}')
					remove_updown_sample_between(dest, source)
				elif dest.resolution > new_resolution:
					print(f'replacing resolution node between {id(dest)} and {id(source)} with upsample to meet new resolution {new_resolution}')
					remove_updown_sample_between(dest, source)
					factor = dest.resolution / new_resolution 
					insert_upsample(dest, source, factor)
				else: # dest.resolution < new_resolution
					print(f'replacing resolution node between {id(dest)} and {id(source)} with downsample to meet new resolution {new_resolution}')
					remove_updown_sample_between(dest, source)
					factor = new_resolution / dest.resolution
					insert_downsample(dest, source, factor)
	for boundary_node in incoming_boundary_nodes:
		graph_node = cg2g[boundary_node]
		parents = get_parents(graph_node)

		for p in parents:
			dests = into_component_dests(p, g2cg, component.nodes)
			for dest in dests:
				source = graph_node
				if source.resolution == new_resolution:
					print(f'removing resolution node between {id(dest)} and {id(source)}')
					remove_updown_sample_between(dest, source)
				elif source.resolution > new_resolution:
					print(f'replacing resolution node between {id(dest)} and {id(source)} with downsample to meet new resolution {new_resolution}')
					remove_updown_sample_between(dest, source)
					factor = source.resolution / new_resolution 
					insert_downsample(dest, source, factor)
				else: # source.resolution < new_resolution
					print(f'replacing resolution node between {id(dest)} and {id(source)} with upsample to meet new resolution {new_resolution}')
					remove_updown_sample_between(dest, source)
					factor = new_resolution / source.resolution
					insert_upsample(dest, source, factor)

	G.compute_input_output_channels()
	compute_resolution(G)
	

"""
given a graph G, if there are any adjacent U->U or D->D, merges them into one
upsample or downsample
"""
def normalize(G, parent):
	children = get_children(G)

	if isinstance(G, Upsample): 
		for c in children:
			if isinstance(c, Upsample):
				new_factor = c.factor * G.factor
				dest = G
				source = c.child
				remove_updown_sample_between(dest, source)
				dest = parent 
				remove_updown_sample_between(dest, source)
				insert_upsample(dest, source, new_factor)	
				for source_child in get_children(source):
					normalize(source_child, source)
			else:
				normalize(c, G)
	elif isinstance(G, Downsample):
		for c in children:
			if isinstance(c, Downsample):
				new_factor = c.factor * G.factor
				dest = G
				source = c.child
				remove_updown_sample_between(dest, source)
				dest = parent 
				remove_updown_sample_between(dest, source)
				insert_downsample(dest, source, new_factor)	
				for source_child in get_children(source):
					normalize(source_child, source)
			else:
				normalize(c, G)
	else:
		for c in children:
			normalize(c, G)
"""
Collapses any degenerate components
"""





	
