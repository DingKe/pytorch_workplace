import torch
from torch.autograd import Function


class ClipGradientF(Function):
    def __init__(self, min_val, max_val):
        super(ClipGradientF, self).__init__()

        self.min_val = min_val
        self.max_val = max_val
    
    def forward(self, input):
        output = input.clone()
        return output 

    def backward(self, grad_output):
        grad_input = grad_output.clamp(self.min_val, self.max_val)
        return grad_input
        
# function interfaces
def clip_gradient(x, min_val, max_val):
    clip = ClipGradientF(min_val, max_val)
    return clip(x)
