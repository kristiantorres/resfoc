import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchcifar10net import Net
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(),
      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='/scr1/joseph29/',train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/scr1/joseph29/',train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
  img = img/2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)))
  plt.show()

# Get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Build the network
net = Net()

# Optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Get the GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

print(device)

net.to(device)

for epoch in range(2):

  running_loss = 0.0
  for i,data in enumerate(trainloader, 0):
    # Get the inputs, data is a list of [inputs, labels]
    inputs,labels = data[0].to(device), data[1].to(device)

    # Zero the parameters gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Print statistics
    running_loss += loss.item()
    if i % 2000 == 1999: # print every 2000 mini-batches
      print('[%d,%5d] lss: %.3f' %
          (epoch + 1, i+1, running_loss/2000))
      running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), "/scr1/joseph29/ficar_net.pth")


