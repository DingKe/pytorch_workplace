import torch
import torch.nn as nn
from torch.autograd import Function


class BinarizeF(Function):

    def forward(self, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
