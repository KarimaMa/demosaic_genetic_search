import torch
from graphviz import Digraph


dot = Digraph(name='programtree')
dot.node('mul', 'mul')
dot.node('softmax', 'softmax')
dot.node('conv_f', 'conv_f')
dot.node('conv_w', 'conv_w')

dot.edges([('mul','softmax'), ('mul','conv_f'),('softmax','conv_w')])
dot.render()

