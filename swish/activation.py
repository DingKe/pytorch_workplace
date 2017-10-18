import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, input):
        return 1.67653251702 * input * F.sigmoid(input)
