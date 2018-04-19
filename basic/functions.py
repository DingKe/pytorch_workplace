from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.autograd import Function


class ReLUF(Function):

    @staticmethod
    def forward(cxt, input):
        cxt.save_for_backward(input)

        output = input.clamp(min=0)

        return output

    @staticmethod
    def backward(cxt, grad_output):
        input, = cxt.saved_tensors

        grad_input = grad_output.clone()
        grad_input[input < 0] = 0

        return grad_input


class LinearF(Function):

    @staticmethod
    def forward(cxt, input, weight, bias=None):
        cxt.save_for_backward(input, weight, bias)

        output = input.mm(weight.t())
        if bias is not None:
            output += bias

        return output

    @staticmethod
    def backward(cxt, grad_output):
        input, weight, bias = cxt.saved_variables

        grad_input = grad_weight = grad_bias = None
        if cxt.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if cxt.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and cxt.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        if bias is not None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight


# aliases
relu = ReLUF.apply
linear = LinearF.apply


# simple test
if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1111)
    a = torch.randn(2, 3)

    va = Variable(a, requires_grad=True)
    vb = relu(va)
    print(va.data, vb.data)

    vb.backward(torch.ones(va.size()))
    print(va.grad.data)
