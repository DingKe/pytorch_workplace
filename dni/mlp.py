import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# Hyper Parameters
input_size = 784
hidden_size = 256
dni_size = 1024
num_classes = 10
num_epochs = 50
batch_size = 500
learning_rate = 1e-3

use_cuda = torch.cuda.is_available()

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='../data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class DNI(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DNI, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.fc2(out)
        return out

    def reset_parameters(self):
        super(DNI, self).reset_parameters()
        for param in self.fc2.parameters():
            param.data.zero_()


dni = DNI(hidden_size, dni_size)


class Net1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net1, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.ReLU())

    def forward(self, x):
        return self.mlp.forward(x)


net1 = Net1(input_size, hidden_size)


class Net2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net2, self).__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module('fc1', nn.Linear(input_size, hidden_size))
        self.mlp.add_module('bn1', nn.BatchNorm1d(hidden_size))
        self.mlp.add_module('act1', nn.ReLU())

        self.mlp.add_module('fc', nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        return self.mlp.forward(x)


net2 = Net2(hidden_size, hidden_size, num_classes)


# Loss
xent = nn.CrossEntropyLoss()
mse = nn.MSELoss()

# Optimizers
opt_net1 = torch.optim.Adam(net1.parameters(), lr=learning_rate)
opt_net2 = torch.optim.Adam(net2.parameters(), lr=learning_rate)
opt_dni = torch.optim.Adam(dni.parameters(), lr=learning_rate)

if use_cuda:
    net1.cuda()
    net2.cuda()
    dni.cuda()

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        opt_net1.zero_grad()  # zero the gradient buffer
        opt_net2.zero_grad()  # zero the gradient buffer
        opt_dni.zero_grad()  # zero the gradient buffer

        # Forward, Stage1
        h = net1(images)
        h1 = Variable(h.data, requires_grad=True)
        h2 = Variable(h.data, requires_grad=False)

        # Forward, Stage2
        outputs = net2(h1)

        # Backward
        loss = xent(outputs, labels)
        loss.backward()

        # Synthetic gradient and backward
        grad = dni(h2)
        h.backward(grad)

        # regress
        regress_loss = mse(grad, Variable(h1.grad.data))
        regress_loss.backward()

        # optimize
        opt_net1.step()
        opt_net2.step()
        opt_dni.step()

        if (i + 1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    if use_cuda:
        images = images.cuda()
        labels = labels.cuda()
    images = Variable(images.view(-1, 28 * 28))
    outputs = net2(net1(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))
