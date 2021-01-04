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