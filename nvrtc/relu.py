from __future__ import print_function

import math
import torch
from torch.autograd import Variable
from collections import namedtuple

from cupy.cuda import function
from pynvrtc.compiler import Program

###

kernel = '''
extern "C"
__global__ void relu_forward(float *output, const float *input, int num)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < num; tid += stride) {
     output[tid] = input[tid] >= 0 ? input[tid] : 0;
  }
}

extern "C"
__global__ void relu_backward(float *input_grad, const float *output_grad, const float *input, int num)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < num; tid += stride) {
     input_grad[tid] = input[tid] >= 0 ? output_grad[tid] : 0;
  }
}
'''

###

class GPUReLUF(torch.autograd.Function):
    configured_gpus = {}
    ptx = None

    def compile(self):
        if self.ptx is None:
            program = Program(kernel, 'relu.cu')
            GPUReLUF.ptx = program.compile()

        if torch.cuda.current_device() not in GPUReLUF.configured_gpus:
            m = function.Module()
            m.load(bytes(self.ptx))

            self.relu_forward = m.get_function('relu_forward')
            self.relu_backward = m.get_function('relu_backward')

            Stream = namedtuple('Stream', ['ptr'])
            self.stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

            GPUReLUF.configured_gpus[torch.cuda.current_device()] = (self.relu_forward, self.relu_backward, self.stream)

        self.relu_forward, self.relu_backward, self.stream = GPUReLUF.configured_gpus[torch.cuda.current_device()]

    def forward(self, x):
        self.compile()

        self.save_for_backward(x)

        y = x.new(*x.size())

        ###
        batch_size, hidden_size = x.size()
	num = batch_size * hidden_size
        grid_hidden_size = min(num, 512)
        grid = (int(math.ceil(num / grid_hidden_size)), 1)
        self.relu_forward(grid=grid, block=(grid_hidden_size, 1), args=[y.data_ptr(), x.data_ptr(), num], stream=self.stream)

        return y 

    def backward(self, grad_y):
        self.compile()

        x, = self.saved_tensors

        grad_x = x.new(*x.size())

        ###
        batch_size, hidden_size = x.size()
	num = batch_size * hidden_size
        grid_hidden_size = min(num, 512)
        grid = (int(math.ceil(num/ grid_hidden_size)), 1)
        self.relu_backward(grid=grid, block=(grid_hidden_size, 1), args=[grad_x.data_ptr(), grad_y.data_ptr(), x.data_ptr(), num], stream=self.stream)

        return grad_x


class ReLU(torch.nn.Module):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x, use_cuda=True):
        # Use CUDA by default unless it's available
        use_cuda = use_cuda and torch.cuda.is_available()
        # Ensure the user is aware when ForgetMult is not GPU version as it's far faster
        if use_cuda: assert x.is_cuda, 'GPU ReLU with fast element-wise CUDA kernel requested but tensors not on GPU'
        ###
        return GPUReLUF()(x) if use_cuda else torch.nn.functional.relu(x)


if __name__ == '__main__':
    batch, hidden = 20, 650
    x = Variable(torch.rand(batch, hidden).cuda(), requires_grad=True)
    y = Variable(torch.rand(batch, hidden).cuda(), requires_grad=True)

    print('CUDA ReLU')
    print('=-=-' * 5)

    ya = ReLU()(x, use_cuda=True)
    loss = ya.mean()
    loss.backward()

    print('Result =', loss.data[0])
    print('X grad =', x.grad.mean().data[0])

    x_grad_copy = x.grad.clone()

    print('CPU ReLU')
    print('=-=-' * 5)

    x.cpu()

    x.grad.data *= 0

    yb = ReLU()(x, use_cuda=False)
    print(yb.size())
    loss = yb.mean()
    loss.backward()

    print('Result =', loss.data[0])
    print('X grad =', x.grad.mean().data[0])

    ###

    print()
    print('=-=-' * 5)
    print('(Xgrad - Xgrad).sum() =', (x_grad_copy - x.grad).sum().data[0])
    print('Residual error for result')
    print('=-=-' * 5)
    residual = (ya- yb)
    print(residual.abs().sum().data[0])
