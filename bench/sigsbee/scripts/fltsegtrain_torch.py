import torch
import numpy as np
from sigsbee_fltsegdata import SigsbeeFltSegData
from torch.utils.data import DataLoader
from deeplearn.utils import plotseglabel, torchprogress
from deeplearn.torchnets import Unet
from deeplearn.torchlosses import bal_ce
from genutils.ptyprint import create_inttag

sig_fltseg = SigsbeeFltSegData('/net/thing/scr2/joseph29/sigsbee_fltseg.h5')

dataloader = DataLoader(sig_fltseg,batch_size=20,shuffle=True,num_workers=0)

#for i in range(len(sig_fltseg)):
#  sample = sig_fltseg[i]
#  print(i,sample['img'].size(),sample['lbl'].size())
#  plotseglabel(sample['img'][0].numpy(),sample['lbl'][0].numpy(),show=True,interpolation='bilinear',aratio=0.5)

# Get the GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Get the network
net = Unet()
net.to(device)

# Loss function
criterion = bal_ce(device)
criterion.to(device)

# Optimizer
lr = 1e-4
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

# Training
nepoch = 10

for epoch in range(nepoch):
  print("Epoch: %s/%d"%(create_inttag(epoch,nepoch),nepoch))
  running_loss = 0.0
  for i,data in enumerate(dataloader,0):
    inputs,labels = data['img'].to(device), data['lbl'].to(device)
    #plotseglabel(inputs.numpy()[0,0],labels.numpy()[0,0],show=True,interpolation='bilinear',aratio=0.5)

    # Zero the parameters gradients
    optimizer.zero_grad()

    # Forward prediction and compute loss
    outputs = net(inputs)
    loss = criterion(outputs,labels)
    # Compute gradient
    loss.backward()
    # Update parameters
    optimizer.step()

    # Print statistics
    running_loss += loss.item()
    cor = ((outputs > 0.5).float() == labels).float().sum().item()
    acc = (cor/np.prod(labels.size()))
    if(i%100 == 0):
      torchprogress(i,len(dataloader),running_loss,acc)

  torchprogress(i+1,len(dataloader),running_loss,acc)

torch.save(net.state_dict(), "/scr1/joseph29/sigsbee_fltseg_torch.pth")

