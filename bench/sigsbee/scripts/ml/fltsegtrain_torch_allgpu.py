import torch
import numpy as np
from sigsbee_fltsegdata_gpu import SigsbeeFltSegDataGPU
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from deeplearn.utils import plotseglabel, torchprogress
from deeplearn.torchnets import Unet
from deeplearn.torchlosses import bal_ce
from genutils.ptyprint import create_inttag

# Get the GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Training set
#sig_fltseg = SigsbeeFltSegDataGPU('/net/thing/scr2/joseph29/sigsbee_fltseg.h5',device,verb=True)
sig_fltseg = SigsbeeFltSegDataGPU('/net/thing/scr2/joseph29/sigsbee_focseg.h5',device,verb=True)

# Get idxs and cutoff
nex   = len(sig_fltseg); split = 0.2
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

trloader = DataLoader(sig_fltseg,batch_size=20,num_workers=0,sampler=trsampler)
valoader = DataLoader(sig_fltseg,batch_size=20,num_workers=0,sampler=vasampler)

#for i in range(len(sig_fltseg)):
#  sample = sig_fltseg[i]
#  print(i,sample['img'].size(),sample['lbl'].size())
#  plotseglabel(sample['img'][0].cpu().numpy(),sample['lbl'][0].cpu().numpy(),show=True,interpolation='bilinear',aratio=0.5)

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
  print("Epoch: %s/%d"%(create_inttag(epoch+1,nepoch),nepoch))
  running_loss = 0.0
  for i,trdat in enumerate(trloader,0):
    inputs,labels = trdat['img'], trdat['lbl']

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
    # TODO: save loss and accuracy metrics with epoch
    running_loss += loss.item()
    cor = ((outputs > 0.5).float() == labels).float().sum().item()
    acc = (cor/np.prod(labels.size()))
    if(i%10 == 0):
      torchprogress(i,len(trloader),running_loss,acc)

  torchprogress(i+1,len(trloader),running_loss,acc)

  # Check on validation data
  with torch.no_grad():
    va_loss = 0
    for vadat in valoader:
      # Get example
      vadat,valbl = vadat['img'].to(device), vadat['lbl'].to(device)
      # Make prediction
      vaprd = net(vadat)
      vloss = criterion(vaprd,valbl)
      # Compute loss and accuracy
      va_loss += vloss.item()
      vcor = ((vaprd > 0.5).float() == valbl).float().sum().item()
      vacc = (vcor/np.prod(valbl.size()))
    print("val_loss=%.4g val_acc=%.4f"%(va_loss/len(valoader),vacc))

  #TODO: if the validation loss has decreased, write the network

torch.save(net.state_dict(), "/scr1/joseph29/sigsbee_fltseg_torch_bigagc.pth")

