import sys
sys.path.append(sys.path[0].split("/")[0])
from model_lib import multires_green_model
import argparse
import random
import numpy as np
from mutate import Mutator
from demosaic_ast import *

parser = argparse.ArgumentParser("Demosaic")
parser.add_argument('--default_channels', type=int, default=16, help='num of output channels for conv layers')
parser.add_argument('--max_nodes', type=int, default=33, help='max number of nodes in a tree')
parser.add_argument('--min_subtree_size', type=int, default=2, help='minimum size of subtree in insertion')
parser.add_argument('--max_subtree_size', type=int, default=11, help='maximum size of subtree in insertion')
parser.add_argument('--structural_sim_reject', type=float, default=0.66, help='rejection probability threshold for structurally similar trees')
parser.add_argument('--model_path', type=str, default='models', help='path to save the models')
parser.add_argument('--model_database_dir', type=str, default='model_database', help='path to save model statistics')
parser.add_argument('--database_save_freq', type=int, default=5, help='model database save frequency')
parser.add_argument('--save', type=str, default='SEARCH_MODELS', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--generations', type=int, default=20, help='model search generations')
parser.add_argument('--seed_model_file', type=str, help='')
parser.add_argument('--cost_tiers', type=str, help='list of tuples of cost tier ranges')
parser.add_argument('--tier_size', type=int, default=20, help='how many models to keep per tier')
parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
parser.add_argument('--mutation_failure_threshold', type=int, default=500, help='max number of tries to mutate a tree')
parser.add_argument('--delete_failure_threshold', type=int, default=25, help='max number of tries to find a node to delete')
parser.add_argument('--subtree_selection_tries', type=int, default=50, help='max number of tries to find a subtree when inserting a binary op')
parser.add_argument('--select_insert_loc_tries', type=int, default=8, help='max number of tries to find a insert location for a partner op')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


model_inputs = set(("Input(Bayer)",))

# seed 5 tests invalid spatial resolutions
mutator = Mutator(args, None)
model = multires_green_model()
print(model.dump())

print("---- testing insert LogSub ---")
copy_model = copy_subtree(model)
print("the starting model")
print(copy_model.dump())
insert_tree = mutator.insert_mutation(copy_model, model_inputs, insert_above_node_id=14, insert_op=LogSub)
print("the mutated tree")
print(insert_tree.dump())


print("---- saving model to file ----")
save_ast(insert_tree, "foo.model_ast")
loaded_tree = load_ast("foo.model_ast")

print("reloaded tree")
print(loaded_tree.dump())
loaded_tree.compute_size(set(), count_input_exprs=True)
print("---- testing that the right child of LogSub and AddExp are the same node in the reloaded tree ----")
preorder = loaded_tree.preorder()
for i, n in enumerate(preorder):
	print(f"{i} {n} {n.size}")
	if "1x1" in n.name and n.size == 6:
		print(n.dump())

torch_tree = loaded_tree.ast_to_model()
for n,p in torch_tree.named_parameters():
	print(n)









