import math

import torch
import torch.nn as nn
from torch.autograd import Variable


def kl_loss(mu, logsigma):
    loss = logsigma + (1 + mu ** 2) / (2 * logsigma.exp() ** 2) - 0.5
    return loss.sum()

def log_gaussian(x, mu, logsigma):
    loss = -logsigma
    loss -= (x - mu) ** 2 /  logsigma.exp() ** 2 / 2
    return loss.sum()


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, mode='kl'):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.mode = mode

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logsigma = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_logsigma = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_logsigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Glorot Initialization
        """
        stdv = math.sqrt(2. / (sum(self.weight_mu.size())))
        self.weight_mu.data.normal_(0, stdv)
        self.weight_logsigma.data.normal_(-3, 0.01)
        if self.bias:
            self.bias_mu.data.zero_()
            self.bias_logsigma.data.normal_(-3, 0.01)

    def forward(self, input, deterministic=False):
        if deterministic:
            weight = self.weight_mu
            bias = self.bias_mu
        else:
            eps = Variable(torch.randn(self.weight_mu.size()))
            weight = self.weight_mu + self.weight_logsigma.exp() * eps

            if self.bias:
                eps = Variable(torch.randn(self.bias_mu.size()))
                bias = self.bias_mu + self.bias_logsigma.exp() * eps

        if self.bias is None:
            output = self._backend.Linear()(input, weight)
        else:
            output = self._backend.Linear()(input, weight, bias)

        if self.mode == 'kl':
            aux_loss = kl_loss(self.weight_mu, self.weight_logsigma)
            if self.bias:
                aux_loss += kl_loss(self.bias_mu, self.bias_logsigma)
        elif self.mode == 'variational':
            zeros = Variable(torch.zeros(weight.size()))
            aux_loss = log_gaussian(weight, self.weight_mu, self.weight_logsigma) - log_gaussian(weight, zeros, zeros)
            if self.bias:
                zeros = Variable(torch.zeros(bias.size()))
                aux_loss += log_gaussian(bias, self.bias_mu, self.bias_logsigma) - log_gaussian(bias, zeros, zeros)

        return output, aux_loss

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
