"""
lowered representation of AST to pytorch model
"""

import torch 
import torch.nn as nn
from demosaic_ast import *
from config import IMG_H, IMG_W
from util import extclass

cuda = False

class InputOp(nn.Module):
  def __init__(self):
    super(InputOp, self).__init__()

  def _initialize_parameters(self):
    pass

  def foward(self, bayer):
    return bayer

  def run(self, bayer):
    return bayer
    #return self.forward(bayer)



class AddOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(AddOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def run(self, bayer):
    loperand = self._operands[0].run(bayer)
    roperand = self._operands[1].run(bayer)
    return self.forward(loperand, roperand)

  def forward(self, x, y):
    return x + y

class SubOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(SubOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def run(self, bayer):
    loperand = self._operands[0].run(bayer)
    roperand = self._operands[1].run(bayer)
    return self.forward(loperand, roperand)

  def forward(self, x, y):
    return x - y

class MulOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(MulOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def run(self, bayer):
    loperand = self._operands[0].run(bayer)
    roperand = self._operands[1].run(bayer)
    return self.forward(loperand, roperand)

  def forward(self, x, y):
    return x * y

class LogSubOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(LogSubOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def run(self, bayer):
    loperand = self._operands[0].run(bayer)
    roperand = self._operands[1].run(bayer)
    return self.forward(loperand, roperand)

  def forward(self, x, y):
    return torch.log(x) - torch.log(y)

class AddExpOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(AddExpOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def run(self, bayer):
    loperand = self._operands[0].run(bayer)
    roperand = self._operands[1].run(bayer)
    return self.forward(loperand, roperand)

  def forward(self, x, y):
    return torch.exp(x) + torch.exp(y)

class StackOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(StackOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def run(self, bayer):
    loperand = self._operands[0].run(bayer)
    roperand = self._operands[1].run(bayer)
    return self.forward(loperand, roperand)

  def forward(self, x, y):
    return torch.cat((x, y), 1)

class ChromaExtractorOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(ChromaExtractorOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def run(self, bayer):
    loperand = self._operands[0].run(bayer)
    roperand = self._operands[1].run(bayer)
    return self.forward(loperand, roperand)

  def forward(self, x, y):
    pass

class GreenExtractorOp(nn.Module):
  def __init__(self, loperand, roperand):
    super(GreenExtractorOp, self).__init__()
    self._operands = nn.ModuleList([loperand, roperand])
    self.mask = torch.zeros((IMG_H, IMG_W))
    self.mask[0::2,1::2] = 1
    self.mask[1::2,0::2] = 1
    if cuda:
      self.mask = self.mask.cuda()

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()
    self._operands[1]._initialize_parameters()

  def run(self, bayer):
    loperand = self._operands[0].run(bayer)
    roperand = self._operands[1].run(bayer)
    return self.forward(loperand, roperand)

  def forward(self, x, bayer):
    green = x * self.mask + bayer[:,0,...].unsqueeze(1)
    return green

class SoftmaxOp(nn.Module):
  def __init__(self, operand):
    super(SoftmaxOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.f = nn.Softmax(dim=1)
  
  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def run(self, bayer):
    operand = self._operands[0].run(bayer)
    return self.forward(operand)

  def forward(self, x):
    return self.f(x)

class ReluOp(nn.Module):
  def __init__(self, operand):
    super(ReluOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.f = nn.ReLU()

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def run(self, bayer):
    operand = self._operands[0].run(bayer)
    return self.forward(operand)

  def forward(self, x):
    return self.f(x)

class LogOp(nn.Module):
  def __init__(self, operand):
    super(LogOp, self).__init__()
    self._operands = nn.ModuleList([operand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def run(self, bayer):
    operand = self._operands[0].run(bayer)
    return self.forward(operand)

  def forward(self, x):
    return torch.log(x)

class ExpOp(nn.Module):
  def __init__(self, operand):
    super(ExpOp, self).__init__()
    self._operands = nn.ModuleList([operand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def run(self, bayer):
    operand = self._operands[0].run(bayer)
    return self.forward(operand)

  def forward(self, x):
    return torch.exp(x)

class DownsampleOp(nn.Module):
  def __init__(self, operand, C_in):
    super(DownsampleOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.pool = nn.Conv2d(C_in, C_in, 5, stride=3, padding=(3,3), bias=False)
  
  def _initialize_parameters(self):
    w = 1.0/9.0
    weights = torch.zeros(self.pool.weight.shape) 
    weights[:,:,0::2,0::2] = w
    self.pool.weight.data = weights
    self.pool.weight.requires_grad = False
    self._operands[0]._initialize_parameters()

  def run(self, bayer):
    operand = self._operands[0].run(bayer)
    return self.forward(operand)

  def forward(self, x):
    return self.pool(x)
    
class UpsampleOp(nn.Module):
  def __init__(self, operand, C_in):
    super(UpsampleOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    # extract predicted values centered at Gr, R, B, Gb into a dense quad
    self.flat2dense = nn.Conv2d(C_in, C_in*4, 2, padding=0, stride=2, bias=False)
    # splat them into their corresponding locations in an upsampled bayer
    self.dense2fullres_bayer = nn.ConvTranspose2d(C_in*4, C_in*4, 6, groups=C_in*4, stride=6, bias=False)
    self.in_c = C_in
  
  def _initialize_parameters(self):
    init_flat2dense(self.flat2dense.weight.data)
    init_dense2fullres_bayer(self.dense2fullres_bayer.weight.data)
    self.flat2dense.weight.requires_grad = False
    self.dense2fullres_bayer.weight.requires_grad = False
    self._operands[0]._initialize_parameters()

  def run(self, bayer):
    operand = self._operands[0].run(bayer)
    return self.forward(operand)

  def forward(self, x):
    # 1st channel centered at Gr, 2nd channel centered at R, 3rd channel at B, 4th channel at Gb
    dense = self.flat2dense(x)
    
    # spread Gr, R, B, and Gb centered weights
    fullres = self.dense2fullres_bayer(dense)
    split = torch.split(fullres, self.in_c, dim=1)

    # crop centered values differently
    croppedGr = split[0][:,:,0:-4,0:-4]
    croppedR  = split[1][:,:,0:-4, 4: ]
    croppedB  = split[2][:,:,4: , 0:-4]
    croppedGb = split[3][:,:,4: , 4: ]

    return (croppedGr + croppedR + croppedB + croppedGb)
    
class Conv1x1Op(nn.Module):
  def __init__(self, operand, C_in, C_out):
    super(Conv1x1Op, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.f = nn.Conv2d(C_in, C_out, (1, 1), bias=False, padding=0)

  def _initialize_parameters(self):
    nn.init.xavier_normal_(self.f.weight)
    self._operands[0]._initialize_parameters()

  def run(self, bayer):
    operand = self._operands[0].run(bayer)
    return self.forward(operand)

  def forward(self, x):
    return self.f(x)

# 1D diagonal convolution from top left corner to bottom right corner
class DiagLRConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, padding):
    super(DiagLRConv, self).__init__()
    self.padding = padding
    #self.filter_w = nn.Parameter(torch.zeros(C_out, C_in, kernel_size)).cuda()
    self.filter_w = nn.Parameter(torch.zeros(C_out, C_in, kernel_size))

  def _initialize_parameters(self):
    nn.init.xavier_normal_(self.filter_w)

  def forward(self, x):
    padding = self.padding
    return nn.functional.conv2d(x, torch.diag_embed(self.filter_w), padding=padding)

# 1D diagonal convolution from top right corner to bottom left corner
class DiagRLConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, padding):
    super(DiagRLConv, self).__init__()
    self.padding = padding
    # self.filter_w = nn.Parameter(torch.zeros(C_out, C_in, kernel_size, kernel_size)).cuda()
    # self.mask = torch.zeros(C_out, C_in, kernel_size, kernel_size).cuda()
    self.filter_w = nn.Parameter(torch.zeros(C_out, C_in, kernel_size, kernel_size))
    self.mask = torch.zeros(C_out, C_in, kernel_size, kernel_size)
    if cuda:
      self.mask = self.mask.cuda()

    for i in range(kernel_size):
      self.mask[..., i, kernel_size-i-1] = 1.0

  def _initialize_parameters(self):
    nn.init.xavier_normal_(self.filter_w)

  def forward(self, x):
    return nn.functional.conv2d(x, (self.filter_w * self.mask), padding=self.padding)

class Conv1DOp(nn.Module):
  def __init__(self, operand, C_in, C_out):
    super(Conv1DOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    assert C_out % 4 == 0, "Output channels must be divisible by 4 to use separable conv"
    self.v = nn.Conv2d(C_in, C_out//4, (5, 1), bias=False, padding=(2, 0))
    self.h = nn.Conv2d(C_in, C_out//4, (1, 5), bias=False, padding=(0, 2))
    self.diag1 = DiagLRConv(C_in, C_out//4, 5, 2)
    self.diag2 = DiagRLConv(C_in, C_out//4, 5, 2)

  def _initialize_parameters(self):
    nn.init.xavier_normal_(self.v.weight)
    nn.init.xavier_normal_(self.h.weight)
    self.diag1._initialize_parameters()
    self.diag2._initialize_parameters()
    self._operands[0]._initialize_parameters()

  def run(self, bayer):
    operand = self._operands[0].run(bayer)
    return self.forward(operand)

  def forward(self, x):
    return torch.cat((self.v(x), self.h(x), self.diag1(x), self.diag2(x)), 1)

class Conv2DOp(nn.Module):
  def __init__(self, operand, C_in, C_out):
    super(Conv2DOp, self).__init__()
    self._operands = nn.ModuleList([operand])
    self.f = nn.Conv2d(C_in, C_out, (5,5), bias=False, padding=2)

  def _initialize_parameters(self):
    nn.init.xavier_normal_(self.f.weight)
    self._operands[0]._initialize_parameters()

  def run(self, bayer):
    operand = self._operands[0].run(bayer)
    return self.forward(operand)

  def forward(self, x):
    return self.f(x)

class SumROp(nn.Module):
  def __init__(self, operand):
    super(SumROp, self).__init__()
    self._operands = nn.ModuleList([operand])

  def _initialize_parameters(self):
    self._operands[0]._initialize_parameters()

  def run(self, bayer):
    operand = self._operands[0].run(bayer)
    return self.forward(operand)

  def forward(self, x):
    return torch.sum(x, dim=1).unsqueeze(1)  


@extclass(Input)
def ast_to_model(self):
  return InputOp()

@extclass(Add)
def ast_to_model(self):
  lmodel = self.lchild.ast_to_model()
  rmodel = self.rchild.ast_to_model()
  return AddOp(lmodel, rmodel)

@extclass(Sub)
def ast_to_model(self):
  lmodel = self.lchild.ast_to_model()
  rmodel = self.rchild.ast_to_model()
  return SubOp(lmodel, rmodel)

@extclass(Mul)
def ast_to_model(self):
  lmodel = self.lchild.ast_to_model()

  rmodel = self.rchild.ast_to_model()
  return MulOp(lmodel, rmodel)

@extclass(LogSub)
def ast_to_model(self):
  lmodel = self.lchild.ast_to_model()
  rmodel = self.rchild.ast_to_model()
  return LogSubOp(lmodel, rmodel)

@extclass(AddExp)
def ast_to_model(self):
  lmodel = self.lchild.ast_to_model()
  rmodel = self.rchild.ast_to_model()
  return AddExpOp(lmodel, rmodel)

@extclass(Stack)
def ast_to_model(self):
  lmodel = self.lchild.ast_to_model()
  rmodel = self.rchild.ast_to_model()
  return StackOp(lmodel, rmodel)

@extclass(ChromaExtractor)
def ast_to_model(self):
  lmodel = self.lchild.ast_to_model()
  rmodel = self.rchild.ast_to_model()
  return ChromaExtractorOp(lmodel, rmodel)

@extclass(GreenExtractor)
def ast_to_model(self):
  lmodel = self.lchild.ast_to_model()
  rmodel = self.rchild.ast_to_model()
  return GreenExtractorOp(lmodel, rmodel)

@extclass(Softmax)
def ast_to_model(self):
  child_model = self.child.ast_to_model()
  return SoftmaxOp(child_model)

@extclass(Relu)
def ast_to_model(self):
  child_model = self.child.ast_to_model()
  return ReluOp(child_model)

@extclass(Log)
def ast_to_model(self):
  child_model = self.child.ast_to_model()
  return LogOp(child_model)

@extclass(Exp)
def ast_to_model(self):
  child_model = self.child.ast_to_model()
  return ExpOp(child_model)

@extclass(Downsample)
def ast_to_model(self):
  child_model = self.child.ast_to_model()
  return DownsampleOp(child_model, self.in_c)

@extclass(Upsample)
def ast_to_model(self):
  child_model = self.child.ast_to_model()
  return UpsampleOp(child_model, self.in_c)

@extclass(Conv1x1)
def ast_to_model(self):
  child_model = self.child.ast_to_model()
  return Conv1x1Op(child_model, self.in_c, self.out_c)

@extclass(Conv1D)
def ast_to_model(self):
  child_model = self.child.ast_to_model()
  return Conv1DOp(child_model, self.in_c, self.out_c)

@extclass(Conv2D)
def ast_to_model(self):
  child_model = self.child.ast_to_model()
  return Conv2DOp(child_model, self.in_c, self.out_c)

@extclass(SumR)
def ast_to_model(self):
  child_model = self.child.ast_to_model()
  return SumROp(child_model)
