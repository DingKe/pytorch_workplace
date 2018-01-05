import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from modules import Linear


torch.manual_seed(1111)

# Hyper Parameters 
input_size = 784
hidden_size = 400
num_classes = 10
num_epochs = 200 
batch_size = 100
bias = True
learning_rate = 0.01

scale = 1e-4
mode = 'variational' # 'kl'
deterministic = True
num_samples = 10

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

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = Linear(input_size, hidden_size, bias=bias, mode=mode) 
        self.relu1 = nn.ReLU()
        self.fc2 = Linear(hidden_size, hidden_size, bias=bias, mode=mode) 
        self.relu2 = nn.ReLU()
        self.fc3 = Linear(hidden_size, num_classes, bias=bias, mode=mode)  
    
    def forward(self, x, deterministic=False):
        out, l1 = self.fc1(x, deterministic)
        out = self.relu1(out)
        out, l2 = self.fc2(out, deterministic)
        out = self.relu2(out)
        out, l3 = self.fc3(out, deterministic)
        return out, l1 + l2 + l3
    
net = Net(input_size, hidden_size, num_classes)

    
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

try:
    net.load_state_dict(torch.load('mlp_{}_{}.pkl'.format(mode, 1)))
except:
    pass

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs, complexity_loss  = net(images)
        likely_loss = criterion(outputs, labels)
        loss = scale * complexity_loss + likely_loss
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, likely_loss.data[0]))

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        if deterministic:
            outputs, _ = net(images, deterministic)
            outputs = outputs.data
        else:
            outputs = torch.zeros(images.size(0), 10)
            for i in range(num_samples):
                output, _ = net(images)
                outputs += output.data
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # Save the Model
    torch.save(net.state_dict(), 'mlp_{}_{}.pkl'.format(mode, epoch+1))
