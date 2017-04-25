import math

import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.autograd import Variable

from functions import clip_grad


class RNNCellBase(Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class RNNCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip

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
        output = clip_grad(output, -self.grad_clip, self.grad_clip) # avoid explosive gradient
        output = F.relu(output)

        return output


class GRUCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip

        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        ih = F.linear(input, self.weight_ih, self.bias)
        hh = F.linear(h, self.weight_hh)

        if self.grad_clip:
            ih = clip_grad(ih, -self.grad_clip, self.grad_clip)
            hh = clip_grad(hh, -self.grad_clip, self.grad_clip)

        r = F.sigmoid(ih[:, :self.hidden_size] + hh[:, :self.hidden_size])
        i = F.sigmoid(ih[:, self.hidden_size: self.hidden_size * 2] + hh[:, self.hidden_size: self.hidden_size * 2])
        n = F.relu(ih[:, self.hidden_size * 2:] * r + hh[:, self.hidden_size * 2:])
        h = (1 - i) * n + i * h

        return h


class LSTMCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip

        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h, c = hx

        pre = F.linear(input, self.weight_ih, self.bias) \
                    + F.linear(h, self.weight_hh)

        if self.grad_clip:
            pre = clip_grad(pre, -self.grad_clip, self.grad_clip)

        i = F.sigmoid(pre[:, :self.hidden_size])
        f = F.sigmoid(pre[:, self.hidden_size: self.hidden_size * 2])
        g = F.tanh(pre[:, self.hidden_size * 2: self.hidden_size * 3])
        o = F.sigmoid(pre[:, self.hidden_size * 3:])

        c = f * c + i * g
        h = o * F.tanh(c)
        return h, c


class LSTMPCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, recurrent_size, bias=True, grad_clip=None):
        super(LSTMPCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_size = recurrent_size
        self.grad_clip = grad_clip

        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, recurrent_size))
        self.weight_rec = Parameter(torch.Tensor(recurrent_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h, c = hx

        pre = F.linear(input, self.weight_ih, self.bias) \
                    + F.linear(h, self.weight_hh)

        if self.grad_clip:
            pre = clip_grad(pre, -self.grad_clip, self.grad_clip)

        i = F.sigmoid(pre[:, :self.hidden_size])
        f = F.sigmoid(pre[:, self.hidden_size: self.hidden_size * 2])
        g = F.tanh(pre[:, self.hidden_size * 2: self.hidden_size * 3])
        o = F.sigmoid(pre[:, self.hidden_size * 3:])

        c = f * c + i * g
        h = o * F.tanh(c)
        h = F.linear(h, self.weight_rec)
        return h, c


class RNN(Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 return_sequences=True, grad_clip=None):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_sequences = return_sequences
        self.grad_clip = grad_clip

        self.cell0= RNNCell(input_size, hidden_size, bias, grad_clip=grad_clip)
        for i in range(1, num_layers):
            cell = RNNCell(hidden_size, hidden_size, bias, grad_clip=grad_clip)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.hidden_size))
            initial_states = [zeros] * self.num_layers
        if type(initial_states) not in [list, tuple]:
            initial_states = [initial_states] * self.num_layers
        assert len(initial_states) == self.num_layers

        states = initial_states
        outputs = []

        time_steps = input.size(1)
        for t in range(time_steps):
            x = input[:, t, :]
            for l in range(self.num_layers):
                x = getattr(self, 'cell{}'.format(l))(x, states[l])
                states[l] = x
            outputs.append(x)

        if self.return_sequences:
            output = torch.stack(outputs).transpose(0, 1)
        else:
            output = outputs[-1]
        return output


class GRU(Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 return_sequences=True, grad_clip=None):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_sequences = return_sequences
        self.grad_clip = grad_clip

        self.cell0= GRUCell(input_size, hidden_size, bias, grad_clip=grad_clip)
        for i in range(1, num_layers):
            cell = GRUCell(hidden_size, hidden_size, bias, grad_clip=grad_clip)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.hidden_size))
            initial_states = [zeros] * self.num_layers
        if type(initial_states) not in [list, tuple]:
            initial_states = [initial_states] * self.num_layers
        assert len(initial_states) == self.num_layers

        states = initial_states
        outputs = []

        time_steps = input.size(1)
        for t in range(time_steps):
            x = input[:, t, :]
            for l in range(self.num_layers):
                x = getattr(self, 'cell{}'.format(l))(x, states[l])
                states[l] = x
            outputs.append(x)

        if self.return_sequences:
            output = torch.stack(outputs).transpose(0, 1)
        else:
            output = outputs[-1]
        return output


class LSTM(Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 return_sequences=True, grad_clip=None):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_sequences = return_sequences
        self.grad_clip = grad_clip

        self.cell0= LSTMCell(input_size, hidden_size, bias, grad_clip=grad_clip)
        for i in range(1, num_layers):
            cell = LSTMCell(hidden_size, hidden_size, bias, grad_clip=grad_clip)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.hidden_size))
            initial_states = [(zeros, zeros)] * self.num_layers
        assert len(initial_states) == self.num_layers

        states = initial_states
        outputs = []

        time_steps = input.size(1)
        for t in range(time_steps):
            x = input[:, t, :]
            for l in range(self.num_layers):
                hx = getattr(self, 'cell{}'.format(l))(x, states[l])
                states[l] = hx 
                x = hx[0]
            outputs.append(x)

        if self.return_sequences:
            output = torch.stack(outputs).transpose(0, 1)
        else:
            output = outputs[-1]
        return output


class LSTMP(Module):

    def __init__(self, input_size, hidden_size, recurrent_size, num_layers=1, bias=True, 
                 return_sequences=True, grad_clip=None):
        super(LSTMP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_size = recurrent_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_sequences = return_sequences
        self.grad_clip = grad_clip

        self.cell0= LSTMPCell(input_size, hidden_size, recurrent_size, bias, grad_clip=grad_clip)
        for i in range(1, num_layers):
            cell = LSTMPCell(recurrent_size, hidden_size, recurrent_size, bias, grad_clip=grad_clip)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            zeros_h = Variable(torch.zeros(input.size(0), self.recurrent_size))
            zeros_c = Variable(torch.zeros(input.size(0), self.hidden_size))
            initial_states = [(zeros_h, zeros_c)] * self.num_layers
        assert len(initial_states) == self.num_layers

        states = initial_states
        outputs = []

        time_steps = input.size(1)
        for t in range(time_steps):
            x = input[:, t, :]
            for l in range(self.num_layers):
                hx = getattr(self, 'cell{}'.format(l))(x, states[l])
                states[l] = hx 
                x = hx[0]
            outputs.append(x)

        if self.return_sequences:
            output = torch.stack(outputs).transpose(0, 1)
        else:
            output = outputs[-1]
        return output
