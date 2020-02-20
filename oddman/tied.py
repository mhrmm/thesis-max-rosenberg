import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import math

class tiedLinear(nn.Module):
    def __init__(self, in_features, bias=True):
        super(tiedLinear, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        repeated_weight = self.weight.repeat(self.in_features, 1)
        return F.linear(input, repeated_weight, self.bias)

layer = tiedLinear(4, False)
x = Variable(torch.randn(4))
out = sum(layer(x))
out.backward()
params = next(layer.parameters())
print(x)
print(params.grad)  # will be 4 times x because of the repeated weights. Autograd has taken the repetition into account
