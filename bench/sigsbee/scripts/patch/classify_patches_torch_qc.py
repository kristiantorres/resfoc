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

qc = True
for iex in range(nex):
  deffound = False
  idx = np.random.randint(nex)
  # Focused
  cubf = sig_foc[idx]['img'].numpy().squeeze()
  focl = sig_foc[idx]['lbl'].numpy().squeeze()
  fstk = foc_seg[idx]['img'].unsqueeze(0)
  slbl = foc_seg[idx]['lbl'].numpy().squeeze()
  # Defocused
  cubd = sig_def[idx]['img'].numpy().squeeze()
  defl = sig_def[idx]['lbl'].numpy().squeeze()
  dstk = def_seg[idx]['img'].unsqueeze(0)
  slbl = def_seg[idx]['lbl'].numpy().squeeze()
  # Make a prediction on the stacks
  with torch.no_grad():
    fprd = torch.sigmoid(net(fstk)).numpy().squeeze()
    dprd = torch.sigmoid(net(dstk)).numpy().squeeze()
  # Compute fault metrics
  fltnum = np.sum(slbl)
  print("Example %d Faultpix=%d"%(idx,fltnum))
  corrprb = corrsim(fprd,dprd)
  fpvar = varimax(fprd); dpvar = varimax(dprd)
  pvarrat = dpvar/fpvar
  print("Fpvar=%f Dpvar=%f Corrprb=%f Rat=%f"%(fpvar,dpvar,corrprb,pvarrat))
  # Angle metrics
  fsemb = semblance_power(cubf[31:])
  dsemb = semblance_power(cubd[31:])
  sembrat = dsemb/fsemb
  print("Fsemb=%f Dsemb=%f Rat=%f"%(fsemb,dsemb,sembrat))
  if(sembrat < 0.4):
    print("Defocused Fault/Angle")
    wh5.write_examples(cubd[np.newaxis],flbl)
    deffound = True
    #TODO: don't forget the continue!
    # continue
  # If example has faults, use fault criteria
  if(fltnum > pixthresh and not deffound):
    if(sembrat < thresh1 and pvarrat  < thresh1):
      print("Defocused fault")
      wh5.write_examples(cubd[np.newaxis],flbl)
    elif(sembrat < thresh2 or pvarrat < thresh2):
      print("Defocused fault")
      wh5.write_examples(cubd[np.newaxis],flbl)
    elif(corrprb < thresh1):
      print("Defocused fault")
      wh5.write_examples(cubd[np.newaxis],flbl)
  else:
    # Compute angle metrics
    if(sembrat < thresh3 and not deffound):
      print("Defocused angle")
      wh5.write_examples(cubd[np.newaxis],flbl)
  if(qc and iex%1 == 0):
    plot_cubeiso(cubf[31:],stack=True,elev=15,show=False,verb=False,title='Focused')
    plot_cubeiso(cubd[31:],stack=True,elev=15,show=False,verb=False,title='Defocused')
    fig,axarr = plt.subplots(1,5,figsize=(15,3))
    axarr[0].imshow(fstk.squeeze(),cmap='gray',interpolation='bilinear',aspect='auto')
    axarr[0].set_title('Focused')
    axarr[1].imshow(dstk.squeeze(),cmap='gray',interpolation='bilinear',aspect='auto')
    axarr[1].set_title('Defocused')
    axarr[2].imshow(slbl,cmap='jet',interpolation='none',aspect='auto')
    axarr[3].imshow(fprd,cmap='jet',interpolation='bilinear',aspect='auto')
    axarr[3].set_title('Focused')
    axarr[4].imshow(dprd,cmap='jet',interpolation='bilinear',aspect='auto')
    axarr[4].set_title('Defocused')
    plt.show()
  print("")

