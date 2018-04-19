# functions/relu.py
import torch
from torch.autograd import Function
from _ext import ext_lib


class ReLUF(Function):
    def forward(self, input):
        self.save_for_backward(input)

        output = input.new()
        if not input.is_cuda:
            ext_lib.relu_forward(input, output)
        else:
            raise Exception("No CUDA Implementation")
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors

        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            ext_lib.relu_backward(grad_output, input, grad_input)
        else:
            raise Exception("No CUDA Implementation")
        return grad_input
