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
                                          
