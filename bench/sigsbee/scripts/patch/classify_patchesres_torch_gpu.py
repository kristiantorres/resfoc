import os
import numpy as np
import torch
from bench.sigsbee.scripts.ml.sigsbee_focdata import SigsbeeFocData
from deeplearn.torchnets import Unet
from genutils.ptyprint import progressbar
from genutils.plot import plot_cubeiso
from deeplearn.dataloader import WriteToH5, load_alldata
from deeplearn.focuslabels import corrsim, semblance_power, varimax
import matplotlib.pyplot as plt

# Data loaders
# Stopped at 26997 of 128960
sig_foc = SigsbeeFocData('/net/thing/scr2/joseph29/sigsbee_fltfoc.h5')
sig_res = SigsbeeFocData('/net/thing/scr2/joseph29/sigsbee_fltres.h5')
rstks,_     = load_alldata('/net/thing/scr2/joseph29/sigsbee_resseg.h5',None,1,verb=True)
fstks,slbls = load_alldata('/net/thing/scr2/joseph29/sigsbee_focseg.h5',None,1,verb=True)

nex = rstks.shape[0]

# Torch network
net = Unet()
device = torch.device('cuda:2')
net.load_state_dict(torch.load('/net/jarvis/scr1/joseph29/sigsbee_fltseg_torch_bigagc.pth',map_location=device))
gnet = net.to(device)

# Copy to GPU
lblst  = np.ascontiguousarray(np.transpose(slbls,(0,3,1,2)))
rstkst = np.ascontiguousarray(np.transpose(rstks,(0,3,1,2)))
fstkst = np.ascontiguousarray(np.transpose(fstks,(0,3,1,2)))
trstks = torch.from_numpy(rstkst.astype('float32')).to(device).unsqueeze(1)
tfstks = torch.from_numpy(fstkst.astype('float32')).to(device).unsqueeze(1)

# Output predictions
rprdsg = torch.zeros(list(trstks.size()),dtype=torch.float32,device=device)
fprdsg = torch.zeros(list(tfstks.size()),dtype=torch.float32,device=device)

# Make predictions
with torch.no_grad():
  for iex in progressbar(range(nex),"nex:"):
    rprdsg[iex] = torch.sigmoid(gnet(trstks[iex]))
    fprdsg[iex] = torch.sigmoid(gnet(tfstks[iex]))

rprds  = rprdsg.cpu().numpy().squeeze()
fprds  = fprdsg.cpu().numpy().squeeze()
fstkss = tfstks.cpu().numpy().squeeze()
rstkss = trstks.cpu().numpy().squeeze()
slblss = lblst.squeeze()

pixthresh = 50
thresh1 = 0.8
thresh2 = 0.6
thresh3 = 0.8

wh5 = WriteToH5("/net/thing/scr2/joseph29/sigsbee_fltreslbl_gpu.h5",dsize=1)
flbl = np.zeros([1],dtype='float32')

nex = len(sig_res)

for iex in progressbar(range(nex),"nex:",verb=True):
  # Focused
  cubf = sig_foc[iex]['img'].numpy().squeeze()
  focl = sig_foc[iex]['lbl'].numpy().squeeze()
  fprd = fprds[iex]
  # Defocused
  cubr = sig_res[iex]['img'].numpy().squeeze()
  resl = sig_res[iex]['lbl'].numpy().squeeze()
  rprd = rprds[iex]
  # Compute fault metrics
  slbl    = slblss[iex]
  fltnum  = np.sum(slbl)
  fpvar   = varimax(fprd); rpvar = varimax(rprd)
  pvarrat = rpvar/fpvar
  corrprb = corrsim(fprd,rprd)
  # Angle metrics
  fsemb = semblance_power(cubf[31:])
  rsemb = semblance_power(cubr[31:])
  sembrat = rsemb/fsemb
  if(sembrat < 0.5):
    wh5.write_examples(cubr[np.newaxis],flbl)
    continue
  # If example has faults, use fault criteria
  if(fltnum > pixthresh):
    if(sembrat < thresh1 and pvarrat  < thresh1):
      wh5.write_examples(cubr[np.newaxis],flbl)
    elif(sembrat < thresh2 or pvarrat < thresh2):
      wh5.write_examples(cubr[np.newaxis],flbl)
    elif(corrprb < thresh1):
      wh5.write_examples(cubr[np.newaxis],flbl)
  else:
    # Compute angle metrics
    if(sembrat < thresh3):
      wh5.write_examples(cubr[np.newaxis],flbl)

