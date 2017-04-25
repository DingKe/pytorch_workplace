import torch
from torch.autograd import Function


class ClipGradientF(Function):
    def __init__(self, min, max):
        super(ClipGradientF, self).__init__()

        self.min = min
        self.max = max
    
    def forward(self, input):
        output = input.clone()
        return output 

    def backward(self, grad_output):
        grad_input = grad_output.clamp(self.min, self.max)
        return grad_input
        
# function interfaces
def clip_grad(x, min, max):
    clip = ClipGradientF(min, max)
    return clip(x)
