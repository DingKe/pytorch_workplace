import math

import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.autograd import Variable


def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v


class RNNCellBase(Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class SRUCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, scaels=[0, 0.25, 0.5, 0.9, 0.99], bias=True, grad_clip=None):
        super(RNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.scales = scales 

        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        output = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        if self.grad_clip:
            output = clip_grad(output, -self.grad_clip, self.grad_clip) # avoid explosive gradient
        output = F.relu(output)

        return output


class SRU(Module):

    def __init__(self, input_size, hidden_size, order, num_layers=1, bias=True, 
                 return_sequences=True, grad_clip=None):
        super(SRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_sequences = return_sequences
        self.grad_clip = grad_clip

        kwargs = {'input_size': input_size,
                  'hidden_size': hidden_size,
                  'order': order,
                  'bias': bias,
                  'grad_clip': grad_clip}

        self.cell0= SRUCell(**kwargs)
        for i in range(1, num_layers):
            cell = SRUCell(**kwargs)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.hidden_size))
            initial_states = [zeros] * self.num_layers
        assert len(initial_states) == self.num_layers

        states = initial_states
        outputs = []

        time_steps = input.size(1)
        for t in range(time_steps):
            x = input[:, t, :]
            for l in range(self.num_layers):
                hx = getattr(self, 'cell{}'.format(l))(x, states[l])
                states[l] = hx[0]
                x = hx[1]
            outputs.append(x)

        if self.return_sequences:
            output = torch.stack(outputs).transpose(0, 1)
        else:
            output = outputs[-1]
        return output
