from torch.nn.modules.module import Module
from functions.relu import ReLUF

class ReLUM(Module):
    def forward(self, input):
        return ReLUF()(input)
