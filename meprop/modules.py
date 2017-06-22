import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsify_grad(v, k, simplified=True):
    if simplified:
        v.register_hook(lambda g: simplified_topk(g, k))
    else:
        v.register_hook(lambda g: topk(g, k))
    return v


def simplified_topk(x, k):
    ''' Proof-of-concept implementation of simplified topk
    Note all we neend the k-th largest vaule, thus an algorithm of log(n) complexity exists.
    '''
    original_size = None
    if x.dim() > 2:
        original_size = x.size()
        x = x.view(x.size(0), -1)
    ax = torch.sum(x.data.abs(), 0).view(-1)
    topk, ids = ax.topk(x.size(-1)-k, dim=0, largest=False)
    y = x.clone()
    # zero out small values
    for id in ids:
        y[:, id] = 0

    if original_size:
        y = y.view(original_size)
    return y


def topk(x, k):
    ''' Proof-of-concept implementation of topk.
    '''
    original_size = None
    if x.dim() > 2:
        original_size = x.size()
        x = x.view(x.size(0), -1)
    ax = torch.abs(x.data)
    topk, _ = ax.topk(k)
    topk = topk[:, -1]
    y = x.clone()
    # zero out small values
    y[ax < topk.repeat(x.size(-1), 1).transpose(0, 1)] = 0

    if original_size:
        y = y.view(original_size)
    return y


class meLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, k=1, simplified=True):
        super(meLinear, self).__init__(in_features, out_features, bias)
        self.k = k
        self.simplified = simplified

    def forward(self, input):
        if self.bias is None:
            output = self._backend.Linear()(input, self.weight)
        else:
            output = self._backend.Linear()(input, self.weight, self.bias)
        return sparsify_grad(output, self.k, self.simplified)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()


class meConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, k=1, simplified=True):
        super(meConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                           stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.k = k
        self.simplified = simplified

    def forward(self, input):
        output = F.conv2d(input, self.weight, self.bias, self.stride,
                               self.padding, self.dilation, self.groups)
        return sparsify_grad(output, self.k, self.simplified)

    def reset_parameters(self):
        # Glorot initialization
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v


class meLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None, k=1, simplified=False):
        super(meLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip

        self.k = k
        self.simplified = simplified

        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
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

        pre = sparsify_grad(pre, self.k, self.simplified)

        if self.grad_clip:
            pre = clip_grad(pre, -self.grad_clip, self.grad_clip)

        i = F.sigmoid(pre[:, :self.hidden_size])
        f = F.sigmoid(pre[:, self.hidden_size: self.hidden_size * 2])
        g = F.tanh(pre[:, self.hidden_size * 2: self.hidden_size * 3])
        o = F.sigmoid(pre[:, self.hidden_size * 3:])

        c = f * c + i * g
        h = o * F.tanh(c)
        return h, c


class meLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 return_sequences=True, grad_clip=None, k=1, simplified=False):
        super(meLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_sequences = return_sequences
        self.grad_clip = grad_clip

        self.k = k
        self.simplified = simplified

        kwargs = {'input_size': input_size,
                  'hidden_size': hidden_size,
                  'bias': bias,
                  'grad_clip': grad_clip,
                  'k': k,
                  'simplified': simplified}

        self.cell0= meLSTMCell(**kwargs)
        for i in range(1, num_layers):
            kwargs['input_size'] = hidden_size
            cell = meLSTMCell(**kwargs)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.hidden_size))
            initial_states = [(zeros, zeros), ] * self.num_layers
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
            outputs.append(hx)

        if self.return_sequences:
            hs, cs = zip(*outputs)
            h = torch.stack(hs).transpose(0, 1)
            c = torch.stack(cs).transpose(0, 1)
            output = (h, c)
        else:
            output = outputs[-1]
        return output
