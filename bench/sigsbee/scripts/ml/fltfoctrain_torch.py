import os
import torch
import numpy as np
from sigsbee_focdata import SigsbeeFocData
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from deeplearn.utils import torchprogress
from deeplearn.torchnets import Vgg3_3d
from deeplearn.torchlosses import bal_ce
from torch.nn import BCELoss
from genutils.ptyprint import create_inttag
from genutils.plot import plot_cubeiso
import matplotlib.pyplot as plt

# Get the GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Training set
sig_fdr = SigsbeeFocData('/net/thing/scr2/joseph29/sigsbee_focdefres.h5',nexmax=50000)
sig_tst = SigsbeeFocData('/net/thing/scr2/joseph29/sigsbee_labeled.h5')

# Get idxs and cutoff
nex   = len(sig_fdr); split = 0.2
idxs  = list(range(nex))
split = int(np.floor(split*nex))

# Shuffle the idxs
seed  = 1992
np.random.seed(seed)
np.random.shuffle(idxs)

# Randomly split examples
trnidx,validx = idxs[split:], idxs[:split]

# Samplers
trsampler = SubsetRandomSampler(trnidx)
vasampler = SubsetRandomSampler(validx)

trbsize = 20; vabsize = 20; tsbsize = 1
trloader = DataLoader(sig_fdr,batch_size=trbsize,num_workers=0,sampler=trsampler)
valoader = DataLoader(sig_fdr,batch_size=vabsize,num_workers=0,sampler=vasampler)
tsloader = DataLoader(sig_tst,batch_size=tsbsize,num_workers=0)

#for i in range(len(sig_fdr)):
#  idx = np.random.randint(len(sig_fdr))
#  sample = sig_fdr[idx]
#  print(i,sample['img'].size(),sample['lbl'].size())
#  if(sample['lbl'].item() == 0):
#    print("Defocused")
#  else:
#    print("Focused")
#  print(sample['img'].squeeze().size())
#  plot_cubeiso(sample['img'].squeeze().numpy(),stack=True,elev=15,show=True,verb=False)

# Get the network
net = Vgg3_3d()
net = torch.nn.DataParallel(net,device_ids=[2,3,4,5,6])
net.to(device)

# Loss function
#criterion = bal_ce(device)
criterion = BCELoss()
criterion.to(device)

# Optimizer
lr = 1e-4
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

# Training
nepoch = 250
nprint = 10

for epoch in range(nepoch):
  print("Epoch: %s/%d"%(create_inttag(epoch+1,nepoch),nepoch))
  running_loss = running_corr = 0.0
  for i,trdat in enumerate(trloader,0):
    inputs,labels = trdat['img'].to(device), trdat['lbl'].to(device)

    # Zero the parameters gradients
    optimizer.zero_grad()

    # Forward prediction and compute loss
    #outputs = net(inputs)
    outputs = torch.sigmoid(net(inputs))
    loss = criterion(outputs,labels)
    # Compute gradient
    loss.backward()
    # Update parameters
    optimizer.step()

    # Print statistics
    # TODO: save loss and accuracy metrics with epoch
    running_loss += loss.item()
    running_corr += ((outputs > 0.5).float() == labels).float().sum().item()
    if(i%nprint == 0):
      torchprogress(i,trbsize,len(trloader),running_loss,running_corr)

  torchprogress(i+1,trbsize,len(trloader),running_loss,running_corr)

  # Check on validation data
  with torch.no_grad():
    va_loss = vacc = 0
    # Validation prediction
    for vadat in valoader:
      # Get example
      vaexp,valbl = vadat['img'].to(device), vadat['lbl'].to(device)
      # Make prediction
      #vaprd = net(vaexp)
      vaprd = torch.sigmoid(net(vaexp))
      vloss = criterion(vaprd,valbl)
      # Compute loss and accuracy
      va_loss += vloss.item()
      vacc += ((vaprd > 0.5).float() == valbl).float().sum().item()
    print("val_loss=%.4g val_acc=%.4f"%(va_loss/len(valoader),vacc/(len(valoader)*vabsize)))
    # Sigsbee data prediction
    ts_loss = tacc = 0
    for tsdat in tsloader:
      # Get example
      tsexp,tslbl = tsdat['img'].to(device), tsdat['lbl'].to(device)
      # Make prediction
      #tsprd = net(tsexp)
      tsprd = torch.sigmoid(net(tsexp))
      tloss = criterion(tsprd,tslbl)
      # Compute loss and accuracy
      ts_loss += tloss.item()
      tacc += ((tsprd > 0.5).float() == tslbl).float().sum().item()
    print("tst_loss=%.4g tst_acc=%.4f"%(ts_loss/len(tsloader),tacc/(len(tsloader)*tsbsize)))

# Get the parameter dictionary
try:
  state_dict = net.module.state_dict()
except AttributeError:
  state_dict = net.state_dict()

# Save the parameter dictionary to file
#torch.save(state_dict, "/scr1/joseph29/sigsbee_fltfocbig.pth")

