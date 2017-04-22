import math

import torch
import torch.nn as nn
from torch.autograd import Function


class ReLUF(Function):

    def forward(self, input):
        self.save_for_backward(input)

        output = input.clamp(min=0)

        return output

    def backward(self, grad_output):
        input = self.saved_tensors[0]

        grad_input = grad_output.clone()
        grad_input[input < 0] = 0

        return grad_input


class LinearF(Function):

    def forward(self, input, weight, bias=None):
        self.save_for_backward(input, weight, bias)

        output = torch.mm(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    def backward(self, grad_output):
        input, weight, bias = self.saved_tensors

        grad_input = grad_weight = grad_bias = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(grad_output, weight)
        if self.needs_input_grad[1]:
            grad_weight = torch.mm(grad_output.t(), input)
        if bias is not None and self.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        if bias is not None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight


# simple test
if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1111)
    a = torch.randn(2, 3)

    va = Variable(a, requires_grad=True)
    vb = ReLUF()(va)
    print va.data, vb.data

    vb.backward(torch.ones(va.size()))
    print va.grad.data
