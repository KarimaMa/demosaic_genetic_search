import torch
import torch.nn as nn
from abc import ABC

"""
binop<n,n,n> := Add | Sub | Mul 
binop<m,n,o> := Stack
unop<n,n>    := Softmax | Relu | Log | Exp
unop<m,n>    := Conv1x1 | Conv1D | Conv2D
unop<n,1>    := SumR
"""

K_WIDTH = 5
PAD = 2

class BinopIII(ABC):
    def __init__(self):
        assert False, "Do not try to instantiate abstract expressions"

class BinopIJK(ABC):
    def __init__(self):
        assert False, "Do not try to instantiate abstract expressions"

class UnopII(ABC):
    def __init__(self):
        assert False, "Do not try to instantiate abstract expressions"

class UnopIJ(ABC):
    def __init__(self):
        assert False, "Do not try to instantiate abstract expressions"
        
class UnopI1(ABC):
    def __init__(self):
        assert False, "Do not try to instantiate abstract expressions"

"""----------------------------------------------------------------------"""


class Add(BinopIII):
    def __init__(self):
        pass

    def forward(self, x, y):
        return torch.add(x, y)


class Sub(BinopIII):
    def __init__(self):
        pass

    def forward(self, x, y):
        return x.sub(y)


class Mul(BinopIII):
    def __init__(self):
        pass

    def forward(self, x, y):
        return torch.mul(x, y)


class Stack(BinopIJK):
    def __init__(self):
        pass

    def forward(self, x, y):
        return torch.cat((x, y), 1)


class Softmax(UnopII, nn.Module):
    def __init__(self):
        self.sm = nn.Softmax(dim=1)
    def forward(self,x):
        return self.sm(x)


class Relu(UnopII, nn.Module):
    def __init__(self):
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)
    

class Log(UnopII):
    def __init__(self):
        pass

    def forward(self, x):
        return torch.log(x)


class Exp(UnopII):
    def __init__(self):
        pass

    def forward(self, x):
        return torch.exp(x)


class Conv1x1(UnopIJ, nn.Module):
    def __init__(self, in_c, out_c):
        self.I = in_c
        self.J = out_c
        self.f = nn.Conv2d(in_c, out_c, 1, bias=False, padding=0)

    def forward(self, x):
        return self.f(x)


class Conv1D(UnopIJ, nn.Module):
    def __init__(self, in_c, out_c):
        self.I = in_c
        self.J = out_c

        self.vfilter = nn.Conv2d(in_c, out_c, (K_WIDTH, 1), bias=False, padding=(PAD, 0))
        self.hfilter = nn.Conv2d(in_c, out_c, (1, K_WIDTH), bias=False, padding=(0, PAD))
        self.d1filter_w = nn.Parameter(torch.zeros(out_c, in_c, K_WIDTH ))
        self.d2filter_w = nn.Parameter(torch.zeros(out_c, in_c, K_WIDTH, K_WIDTH)) 
        self.d2mask = torch.zeros(out_c, in_c, K_WIDTH, K_WIDTH)
        for i in range(K_WIDTH):
            self.d2mask[..., i, K_WIDTH-i-1] = 1.0

    def forward(self, x):
        vfilter_output = self.vfilter(x)
        hfilter_output = self.hfilter(x)
        d1filter_output = nn.functional.conv2d(x, torch.diag_embed(self.d1filter_w), padding=PAD)
        d2filter_output = nn.functional.conv2d(x, (self.d2filter_w * self.d2mask), padding=PAD)

        x = torch.cat((d1filter_output, d2filter_output, vfilter_output, hfilter_output), 1)
        return x


class Conv2D(UnopIJ, nn.Module):
    def __init__(self, in_c, out_c):
        self.I = in_c
        self.J = out_c
        
        self.f = nn.Conv2d(in_c, out_c, (K_WIDTH,K_WIDTH), bias=False, padding=(PAD, PAD))

    def forward(self, x):
        return self.f(x)


class SumR(UnopI1):
    def __init__(self):
        pass

    def forward(self, x):
        return torch.sum(x, dim=1)


