import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 64
learning_rate = 1e-3

# dataset
train_dataset = datasets.MNIST(root='./dataset/mnist',
                               train=True,
                               download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((32, 32)), 
                                   transforms.ToTensor()]))
test_dataset = datasets.MNIST(root='./dataset/mnist',
                              train=False,
                              download=True,
                              transform=transforms.Compose([
                                  transforms.Resize((32, 32)), 
                                  transforms.ToTensor()]))

# dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=2,
                                           drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          num_workers=2,
                                          drop_last=True)
                                          
# model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        C1 = nn.Conv2d(1, 6, kernel_size=5)
        S2 = nn.MaxPool2d(kernel_size=2, stride=2)
        C3 = nn.Conv2d(6, 16, kernel_size=5)
        S4 = nn.MaxPool2d(kernel_size=2, stride=2)
        C5 = nn.Conv2d(16, 120, kernel_size=5)
        F6 = nn.Linear(120, 84)
        OUTPUT = nn.Linear(84, 10)

        self.model = nn.Sequential(
            C1,
            S2,
            C3,
            S4,
            C5,
            F6,
            OUTPUT
        )

    def forward(self, x):
        return self.model(x)

# model
model = LeNet5().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


