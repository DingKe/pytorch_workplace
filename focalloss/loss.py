import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        log_input = F.log_softmax(input)
        y = one_hot(target, input.size(-1))

        loss = -1 * y * log_input  # cross entropy
        loss = loss * (1 - log_input) ** self.gamma # focal loss

        return loss.sum()
