from demosaic_ast import *
from tree import Node
from tree_manipulation import get_children


"""
computes the footprint of a DAG
"""


@extclass(Node)
def compute_footprint(self, footprint):
	if type(self) in linear_ops:
		if not isinstance(self, Conv1x1):
			footprint += self.kwidth -1
	elif isinstance(self, Upsample):
		footprint /= self.factor
	elif isinstance(self, Downsample):
		footprint *= self.factor

	children = get_children(self)
	child_footprints = [c.compute_footprint(footprint) for c in children]

	if isinstance(self, Input):
		return footprint / self.resolution

	return max(child_footprints)