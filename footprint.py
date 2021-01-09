from demosaic_ast import *
from tree import Node

"""
computes the footprint of a DAG
"""
@extclass(Node)
def compute_footprint(self):
	footprint = self.compute_footprint_helper(1)
	fullres_footprint = footprint * 2
	return fullres_footprint

@extclass(Node)
def compute_footprint_helper(self, footprint):
	if type(self) is Conv2D or type(self) is Conv1D:
		footprint += (self.kwidth // 2) 
	elif type(self) is Upsample:
		footprint /= 2
	elif type(self) is Downsample:
		footprint = footprint * 2 + 1

	if self.num_children == 3:
		children = [self.child1, self.child2, self.child3]
	elif self.num_children == 2:
		children = [self.lchild, self.rchild]
	elif self.num_children == 1:
		children = [self.child]
	else:
		return footprint

	child_footprints = [c.compute_footprint_helper(footprint) for c in children]
	return max(child_footprints)


"""
Computes the footprint of a subtree assuming a proposed upsample
insertion right above the given node 
"""
def compute_lowres_branch_footprint(node):
	footprint = compute_lowres_branch_footprint_helper(node, 1/2)
	fullres_footprint = footprint * 2
	return fullres_footprint

def compute_lowres_branch_footprint_helper(node, footprint):
	if type(node) is Conv2D or type(node) is Conv1D:
		footprint += (node.kwidth // 2) 
	elif type(node) is Upsample:
		footprint /= 2
	elif type(node) is Downsample:
		footprint = footprint * 2 + 1

	if node.num_children == 3:
		children = [node.child1, node.child2, node.child3]
	elif node.num_children == 2:
		children = [node.lchild, node.rchild]
	elif node.num_children == 1:
		children = [node.child]
	else:
		return footprint

	child_footprints = [compute_lowres_branch_footprint_helper(c, footprint) for c in children]
	return max(child_footprints)


