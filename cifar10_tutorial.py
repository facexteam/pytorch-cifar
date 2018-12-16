#!/usr/bin/env python
# @author: zhaoyafei0210@gmail.com

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as pyplot
import torch
import torchvision
import torchvision.transforms as transforms


USE_GPU = True
device = None

if USE_GPU:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Training on device: ', device)

# 1. Loading and normalizing CIFAR10
train_cbatchsize = 4
train_workers = 2
test_batchsize = 4
test_workers = 2


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform
                                        )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_cbatchsize,
                                          shuffle=True, num_workers=train_workers
                                          )

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform
                                       )
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batchsize,
                                         shuffle=False, num_workers=test_workers
                                         )


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

num_classes = len(classes)

# show some of the training images for fun


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]]
               for j in range(train_cbatchsize)))

# 2. Define a Convolutional Neural Network


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
if USE_GPU:
    net.to(device)

# 3. Define a Loss function and optimizer


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. Train the network

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        if USE_GPU:
            inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #print statistics
        running_loss += loss.item()
        if i % 2000 = 1999:  # print every 2000 mini-batches
            print('[%d, %d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 5. Test the network on the test data

dataiter = iter(testloader)
images, labels = dataiter.next()
if USE_GPU:
    images, labels = images.to(device), labels.to(device)

# print images
imshow(torchvision.utils.make_grids(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]]
                                for j in range(test_batchsize)))

# forward fo the above batch
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(test_batchsize)))

# test the network performance on the whole dataset
correct = 0
total = 0

with torch.no_grad():
    for data in testloader():
        images, labels = data
        if USE_GPU:
            images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100*correct/total))


# Analyze wht are the classes that performed well and the classes that did not perform vell
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        if USE_GPU:
            images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()

        for i in range(test_batchsize):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(num_classes):
    print('Accuracy of %5s: %2d %%' %
          (class[i], 100*class_correct[i]/class_total[i]))
