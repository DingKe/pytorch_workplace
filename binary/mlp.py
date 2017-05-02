import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from modules import BinaryLinear, BinaryTanh
from adam import Adam


torch.manual_seed(1111)


# Hyper Parameters 
input_size = 784
hidden_size = 1024 
num_layers = 1
num_classes = 10
num_epochs = 50
batch_size = 100

lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / num_epochs)

drop_in = 0.2
drop_hid = 0.5

momentum = 0.9
eps = 1e-6

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

# Neural Network Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        self.p_in = nn.Dropout(p=drop_in)
        for i in range(1, self.num_layers+1):
            in_features = input_size if i == 1 else hidden_size
            out_features = hidden_size
            layer = nn.Sequential(
                BinaryLinear(in_features, out_features, bias=False),
                nn.BatchNorm1d(out_features, momentum=momentum, eps=eps),
                BinaryTanh(),
                nn.Dropout(p=drop_hid))
            setattr(self, 'layer{}'.format(i), layer)
        self.fc = BinaryLinear(hidden_size, num_classes, bias=False)  
    
    def forward(self, x):
        out = self.p_in(x)
        for i in range(1, self.num_layers+1):
            out = getattr(self, 'layer{}'.format(i))(out)
        out = self.fc(out)
        return out
    
mlp = MLP(input_size, hidden_size, num_classes, num_layers=num_layers)

    
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = Adam(mlp.parameters(), lr=lr_start)  


def clip_weight(parameters):
    for p in parameters:
        p = p.data
        p.clamp_(-1., 1.)


# learning rate schedule
def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = lr * lr_decay
        param_group['lr'] = lr


# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = mlp(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        clip_weight(mlp.parameters())
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    adjust_learning_rate(optimizer)

    # Test the Model
    mlp.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        outputs = mlp(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    mlp.train()

# Save the Trained Model
torch.save(mlp.state_dict(), 'mlp.pkl')
