import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 10
batch_size = 128
learning_rate = 1e-3

# dataset
train_dataset = datasets.MNIST(root='./dataset/mnist',
                               train=True,
                               download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((32, 32)), 
                                   transforms.ToTensor(),
                                   transforms.Normalize((0,), (1,)),
                               ]))
test_dataset = datasets.MNIST(root='./dataset/mnist',
                              train=False,
                              download=True,
                              transform=transforms.Compose([
                                  transforms.Resize((32, 32)), 
                                  transforms.ToTensor(),
                                  transforms.Normalize((0,), (1,)),
                              ]))

# dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          drop_last=True)
                                          
# model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        C1 = nn.Conv2d(1, 6, 5)
        S2 = nn.MaxPool2d(2, 2)
        C3 = nn.Conv2d(6, 16, 5)
        S4 = nn.MaxPool2d(2, 2)
        C5 = nn.Conv2d(16, 120, 5)
        F6 = nn.Linear(120, 84)
        OUTPUT = nn.Linear(84, 10)

        AV = nn.Tanh()
        self.conv = nn.Sequential(
            C1,
            AV,
            S2,
            C3,
            AV,
            S4,
            C5,
            AV,
        )
        self.fc = nn.Sequential(
            F6,
            AV,
            OUTPUT,
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        conv_output = self.conv(x)
        conv_output_reshaped = conv_output.view(x.size(0), -1)
        fc_output = self.fc(conv_output_reshaped)
        return fc_output

model = LeNet5().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train model
total_step = len(train_loader)
for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training loss
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# test model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'
          .format(100 * correct / total))
