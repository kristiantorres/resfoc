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
sig_foc = SigsbeeFocData('/net/thing/scr2/joseph29/sigsbee_fltfoc.h5')
sig_def = SigsbeeFocData('/net/thing/scr2/joseph29/sigsbee_fltdef.h5')
dstks,slbls = load_alldata('/net/thing/scr2/joseph29/sigsbee_defseg.h5',None,1,verb=True)
fstks,slbls = load_alldata('/net/thing/scr2/joseph29/sigsbee_focseg.h5',None,1,verb=True)

nex = dstks.shape[0]

# Torch network
net = Unet()
device = torch.device('cuda:2')
net.load_state_dict(torch.load('/net/jarvis/scr1/joseph29/sigsbee_fltseg_torch_bigagc.pth',map_location=device))
gnet = net.to(device)

# Copy to GPU
lblst  = np.ascontiguousarray(np.transpose(slbls,(0,3,1,2)))
dstkst = np.ascontiguousarray(np.transpose(dstks,(0,3,1,2)))
fstkst = np.ascontiguousarray(np.transpose(fstks,(0,3,1,2)))
tdstks = torch.from_numpy(dstkst.astype('float32')).to(device).unsqueeze(1)
tfstks = torch.from_numpy(fstkst.astype('float32')).to(device).unsqueeze(1)

# Output predictions
dprdsg = torch.zeros(list(tdstks.size()),dtype=torch.float32,device=device)
fprdsg = torch.zeros(list(tfstks.size()),dtype=torch.float32,device=device)

# Make predictions
with torch.no_grad():
  begex = 0; endex = 20
  for iex in progressbar(range(nex),"nex:"):
    dprdsg[begex:endex] = torch.sigmoid(gnet(tdstks[begex:endex]))
    fprdsg[begex:endex] = torch.sigmoid(gnet(tfstks[begex:endex]))
    begex = endex; endex += 20

dprds  = dprdsg.cpu().numpy().squeeze()
fprds  = fprdsg.cpu().numpy().squeeze()
fstkss = tfstks.cpu().numpy().squeeze()
dstkss = tdstks.cpu().numpy().squeeze()
slblss = lblst.squeeze()

pixthresh = 50
thresh1 = 0.7
thresh2 = 0.5
thresh3 = 0.7

wh5 = WriteToH5("/net/thing/scr2/joseph29/sigsbee_fltdeflbl_gpu.h5",dsize=1)
flbl = np.zeros([1],dtype='float32')

nex = len(sig_def)

for iex in progressbar(range(nex),"nex:",verb=True):
  # Focused
  cubf = sig_foc[iex]['img'].numpy().squeeze()
  focl = sig_foc[iex]['lbl'].numpy().squeeze()
  fprd = fprds[iex]
  # Defocused
  cubd = sig_def[iex]['img'].numpy().squeeze()
  defl = sig_def[iex]['lbl'].numpy().squeeze()
  dprd = dprds[iex]
  # Compute fault metrics
  slbl    = slblss[iex]
  fltnum  = np.sum(slbl)
  fpvar   = varimax(fprd); dpvar = varimax(dprd)
  pvarrat = dpvar/fpvar
  corrprb = corrsim(fprd,dprd)
  # Angle metrics
  fsemb = semblance_power(cubf[31:])
  dsemb = semblance_power(cubd[31:])
  sembrat = dsemb/fsemb
  if(sembrat < 0.4):
    wh5.write_examples(cubd[np.newaxis],flbl)
    continue
  # If example has faults, use fault criteria
  if(fltnum > pixthresh):
    if(sembrat < thresh1 and pvarrat  < thresh1):
      wh5.write_examples(cubd[np.newaxis],flbl)
    elif(sembrat < thresh2 or pvarrat < thresh2):
      wh5.write_examples(cubd[np.newaxis],flbl)
    elif(corrprb < thresh1):
      wh5.write_examples(cubd[np.newaxis],flbl)
  else:
    # Compute angle metrics
    if(sembrat < thresh3):
      wh5.write_examples(cubd[np.newaxis],flbl)

