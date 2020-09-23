import os
import numpy as np
import torch
from bench.sigsbee.scripts.ml.sigsbee_focdata import SigsbeeFocData
from bench.sigsbee.scripts.ml.sigsbee_fltsegdata import SigsbeeFltSegData
from deeplearn.torchnets import Unet
from genutils.ptyprint import progressbar
from genutils.plot import plot_cubeiso
from deeplearn.dataloader import WriteToH5
from deeplearn.focuslabels import corrsim, semblance_power, varimax
import matplotlib.pyplot as plt

# Data loaders
sig_foc = SigsbeeFocData('/net/thing/scr2/joseph29/sigsbee_fltfoc.h5')
sig_def = SigsbeeFocData('/net/thing/scr2/joseph29/sigsbee_fltdef.h5')
foc_seg = SigsbeeFltSegData('/net/thing/scr2/joseph29/sigsbee_focseg.h5')
def_seg = SigsbeeFltSegData('/net/thing/scr2/joseph29/sigsbee_defseg.h5')

# Torch network
net = Unet()
device = torch.device('cpu')
net.load_state_dict(torch.load('/net/jarvis/scr1/joseph29/sigsbee_fltseg_torch_bigagc.pth',map_location=device))

pixthresh = 50
thresh1 = 0.7
thresh2 = 0.5
thresh3 = 0.7

wh5 = WriteToH5("/net/thing/scr2/joseph29/sigsbee_fltdeflbl.h5",dsize=1)
flbl = np.zeros([1],dtype='float32')

nex = len(sig_def)

for iex in progressbar(range(nex),"nex:",verb=True):
  # Focused
  cubf = sig_foc[iex]['img'].numpy().squeeze()
  focl = sig_foc[iex]['lbl'].numpy().squeeze()
  fstk = foc_seg[iex]['img'].unsqueeze(0)
  slbl = foc_seg[iex]['lbl'].numpy().squeeze()
  # Defocused
  cubd = sig_def[iex]['img'].numpy().squeeze()
  defl = sig_def[iex]['lbl'].numpy().squeeze()
  dstk = def_seg[iex]['img'].unsqueeze(0)
  slbl = def_seg[iex]['lbl'].numpy().squeeze()
  # Make a prediction on the stacks
  with torch.no_grad():
    fprd = torch.sigmoid(net(fstk)).numpy().squeeze()
    dprd = torch.sigmoid(net(dstk)).numpy().squeeze()
  # Compute fault metrics
  fltnum = np.sum(slbl)
  fpvar = varimax(fprd); dpvar = varimax(dprd)
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

