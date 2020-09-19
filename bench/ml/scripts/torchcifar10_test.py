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

# Load in the model
net = Net()
net.load_state_dict(torch.load('/scr1/joseph29/ficar_net.pth'))

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('Grund Truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)
# Get the class with highest probability
_, predicted = torch.max(outputs, 1)

# Exaine a single test example
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
# Use no grad to use less memory
with torch.no_grad():
  for data in testloader:
    # Gives a single batch
    images, labels = data
    # Makes a prediction
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % ( 100 * correct / total))

