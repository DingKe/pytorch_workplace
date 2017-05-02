import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from modules import *


# Hyper Parameters
num_epochs = 50
batch_size = 100

lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / num_epochs)

momentum = 0.9
eps = 1e-6

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='../data/',
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            BinaryConv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16, momentum=momentum, eps=eps),
            nn.MaxPool2d(2),
            BinaryTanh())
        self.layer2 = nn.Sequential(
            BinaryConv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32, momentum=momentum, eps=eps),
            nn.MaxPool2d(2),
            BinaryTanh())
        self.fc = BinaryLinear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
cnn = CNN()


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()

# Optimizer
def scale_lr(parameters, lr):
    param_groups = [] 
    other_params = []
    for param in parameters:
        if hasattr(param, 'lr_scale'):
            g = {'params': [param], 'lr': lr * param.lr_scale}
            param_groups.append(g)
        else:
            other_params.append(param)
    param_groups.append({'params': other_params})
    return param_groups

optimizer = torch.optim.Adam(scale_lr(cnn.parameters(), lr_start), lr=lr_start)  

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    # Test the Model
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

    cnn.train()

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')
