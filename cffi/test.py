import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.relu import ReLUM


torch.manual_seed(1111)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.relu = ReLUM()

    def forward(self, input):
        return self.relu(input)

model = MyNetwork()
x = torch.randn(1, 25).view(5, 5)
input = Variable(x, requires_grad=True)
output = model(input)
print(output)
print(input.clamp(min=0))

output.backward(torch.ones(input.size()))
print(input.grad.data)

if torch.cuda.is_available():
    input = input.cuda()
    output = model(input)
    print(output)
    print(input.clamp(min=0))
