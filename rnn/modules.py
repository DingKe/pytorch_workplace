import math

import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.autograd import Variable

from functions import clip_gradient


class RNNCell(Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(hidden_size))
        self.bias_hh = Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        output = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(h, self.weight_hh, self.bias_hh)
        output = clip_gradient(output, -1, 1) # avoid explosive gradient
        output = F.relu(output)

        return output


class RNN(Module):
    def __init__(self, input_size, hidden_size, return_sequences=True):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        self.cell = RNNCell(input_size, hidden_size)

    def forward(self, input, initial_state=None):
        if initial_state is None:
            state = Variable(torch.zeros(input.size(0), self.hidden_size))
        else:
            state = initial_state
        outputs = []

        time_steps = input.size(1)
        for t in range(time_steps):
            state = self.cell(input[:, t, :], state)
            outputs.append(state)

        if self.return_sequences:
            output = torch.stack(outputs)
        else:
            output = outputs[-1]
        return output


class MultiRNN(Module):
    '''Multiple layer RNN
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, return_sequences=True):
        super(MultiRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cell0= RNNCell(input_size, hidden_size)
        for i in range(1, num_layers):
            cell = RNNCell(hidden_size, hidden_size)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            initial_states = [Variable(torch.zeros(input.size(0), self.hidden_size)) for _ in range(self.num_layers)]
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
            output = torch.stack(outputs)
        else:
            output = outputs[-1]
        return output
