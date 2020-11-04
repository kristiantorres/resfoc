import torch
import numpy as np
from hale_focdata_gpu import HaleFocDataGPU
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from deeplearn.utils import torchprogress
from deeplearn.torchnets import Vgg3a_3d, save_torchnet
from torch.nn import BCELoss
from genutils.ptyprint import create_inttag
from genutils.plot import plot_cubeiso
import inpout.seppy as seppy

# Get the GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Training set
bei_fdr = HaleFocDataGPU('/net/thing/scr2/joseph29/halefoc_sm-small.h5',device,verb=True,begex=0,endex=4000)

# Get label shape
npred = bei_fdr[0]['lbl'].shape[0]

# Get idxs and cutoff
nex   = len(bei_fdr); split = 0.2
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

trbsize, vabsize = 20, 20
trloader = DataLoader(bei_fdr,batch_size=trbsize,num_workers=0,sampler=trsampler)
valoader = DataLoader(bei_fdr,batch_size=vabsize,num_workers=0,sampler=vasampler)

#for i in range(len(bei_fdr)):
#  idx = np.random.randint(len(bei_fdr))
#  sample = bei_fdr[idx]
#  print(i,sample['img'].size(),sample['lbl'].size())
#  if(sample['lbl'].item() == 0):
#    print("Defocused")
#  else:
#    print("Focused")
#  plot_cubeiso(sample['img'][0].cpu().numpy(),stack=True,elev=15,show=True,verb=False)

# Get the network
net = Vgg3a_3d()
net.to(device)

# Loss function
criterion = BCELoss()
criterion.to(device)

# Optimizer
lr = 1e-4
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

# Training
nepoch = 10
nprint = 1

# Output metrics
tlss = []; vlss = []
tacc = []; vacc = []

for epoch in range(nepoch):
  print("Epoch: %s/%d"%(create_inttag(epoch+1,nepoch),nepoch))
  running_loss = running_corr = 0.0
  for i,trdat in enumerate(trloader,0):
    inputs,labels = trdat['img'], trdat['lbl']

    # Zero the parameters gradients
    optimizer.zero_grad()

    # Forward prediction and compute loss
    outputs = torch.sigmoid(net(inputs))
    loss = criterion(outputs,labels)
    # Compute gradient
    loss.backward()
    # Update parameters
    optimizer.step()

    # Print statistics
    running_loss += loss.item()
    running_corr += ((outputs > 0.5).float() == labels).float().sum().item()
    if(i%nprint == 0):
      torchprogress(i,trbsize*npred,len(trloader),running_loss,running_corr)

  lv,av = torchprogress(i+1,trbsize*npred,len(trloader),running_loss,running_corr)
  # Save outputs
  tlss.append(lv)
  tacc.append(av)

  # Check on validation data
  with torch.no_grad():
    va_loss = va_acc = 0
    for vadat in valoader:
      # Get example
      vadat,valbl = vadat['img'], vadat['lbl']
      # Make prediction
      vaprd = torch.sigmoid(net(vadat))
      vloss = criterion(vaprd,valbl)
      # Compute loss and accuracy
      va_loss += vloss.item()
      va_acc  += ((vaprd > 0.5).float() == valbl).float().sum().item()
    print("val_loss=%.4g val_acc=%.4f"%(va_loss/len(valoader),va_acc/(len(valoader)*vabsize*npred)))
    # Save outputs
    vlss.append(va_loss/len(valoader))
    vacc.append(va_acc/(len(valoader)*vabsize*npred))

    # Save the net when validation loss decreases
    if(epoch > 0):
      if(vlss[epoch] > vlss[epoch-1]):
        save_torchnet(net,"/scr1/joseph29/hale2_fltfoc.pth")

    #TODO: label some test patches and read in the test dataset

# Save the losses and accuracies
sep = seppy.sep()
sep.write_file("hale_foctloss.H",np.asarray(tlss))
sep.write_file("hale_focvloss.H",np.asarray(vlss))
sep.write_file("hale_foctacc.H",np.asarray(tacc))
sep.write_file("hale_focvacc.H",np.asarray(vacc))

