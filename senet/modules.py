import math

import torch
import torch.nn as nn

class SEWrapper(nn.Module):

    def __init__(self, channels, ratio=4):
        super(SEWrapper, self).__init__()

        self.linear = nn.Sequential(nn.Linear(channels, channels // ratio), 
                                    nn.ReLU(),
                                    nn.Linear(channels // ratio, channels), 
                                    nn.Sigmoid())

    def forward(self, input):
        sq = input.mean(-1).mean(-1)
        ex = self.linear(sq)

        return input * ex.unsqueeze(-1).unsqueeze(-1)
