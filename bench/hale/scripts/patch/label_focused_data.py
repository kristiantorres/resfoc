import os,glob
import inpout.seppy as seppy
import numpy as np
import subprocess
import torch
from deeplearn.torchnets import Unet
from deeplearn.utils import normextract, plot_seglabel, plot_segprobs, plot_patchgrid2d
from deeplearn.torchpredict import segmentfaults
from deeplearn.focuslabels import semblance_power
from genutils.ptyprint import create_inttag
from genutils.plot import plot_img2d, plot_cubeiso
from genutils.movie import viewcube3d

idir = './dat/split_angs/'
nums = list(range(678))

fapre = "fimg"; fspre = "fstk-"
dapre = "dimg"; dspre = "dstk-"
rapre = "rimg"; rspre = "rstk-"
agc = False
if(agc):
  fapre += 'g-'
  dapre += 'g-'
  rapre += 'g-'
else:
  fapre += 'n-'
  dapre += 'n-'
  rapre += 'n-'

idx = np.random.randint(678)
smooth = True
tag = create_inttag(nums[idx],10000)
if(smooth):
  ext = '-sos.H'
else:
  ext = '.H'
fafile = idir + fapre + tag + ext; fsfile = idir + fspre + tag + ext
dafile = idir + dapre + tag + ext; dsfile = idir + dspre + tag + ext
rafile = idir + rapre + tag + ext; rsfile = idir + rspre + tag + ext
lbfile = idir + 'lbl-' + tag + '.H'

# Read in the the neural network
net = Unet()
device = torch.device('cpu')
net.load_state_dict(torch.load('/net/jarvis/scr1/joseph29/hale2_fltsegsm.pth',map_location=device))

# Read in the files for the example
sep = seppy.sep()
faxes,fang = sep.read_file(fafile)
fang = fang.reshape(faxes.n,order='F')
#viewcube3d(fang,width3=1.0,show=False)

faxes,fstk = sep.read_file(fsfile)
fstk = np.ascontiguousarray(fstk.reshape(faxes.n,order='F'))
#plot_patchgrid2d(fstk,nxp=64,nzp=64)
#plot_img2d(fstk,show=False)

daxes,dang = sep.read_file(dafile)
dang = dang.reshape(daxes.n,order='F')
#viewcube3d(dang,width3=1.0,show=False)

daxes,dstk = sep.read_file(dsfile)
dstk = np.ascontiguousarray(dstk.reshape(daxes.n,order='F'))
#plot_img2d(dstk,show=False)

raxes,rang = sep.read_file(rafile)
rang = rang.reshape(raxes.n,order='F')
#viewcube3d(rang,width3=1.0,show=True)

raxes,rstk = sep.read_file(rsfile)
rstk = np.ascontiguousarray(rstk.reshape(raxes.n,order='F'))
#plot_img2d(rstk,show=True)

laxes,lbl = sep.read_file(lbfile)
lbl = lbl.reshape(laxes.n,order='F')
#plot_seglabel(fstk,lbl)

# Segment each of the images
#fprd = segmentfaults(fstk,net)
#dprd = segmentfaults(dstk,net)
#rprd = segmentfaults(rstk,net)
#plot_segprobs(fstk,fprd,show=False)
#plot_segprobs(dstk,dprd,show=False)
#plot_segprobs(rstk,rprd,show=True)

## Patch the example
fangt = np.transpose(fang.T,(0,2,1))
fptches = normextract(fangt,norm=True)

dangt = np.transpose(dang.T,(0,2,1))
dptches = normextract(dangt,norm=True)

rangt = np.transpose(rang.T,(0,2,1))
rptches = normextract(rangt,norm=True)

lptches = normextract(lbl,norm=False)
nptch = lptches.shape[0]

sptches = normextract(fstk,norm=True)

pixthresh = 75

# Loop over each patch
for iptch in range(nptch):
  # Make sure the patch has a fault
  fltnum = np.sum(lptches[iptch])
  if(fltnum > pixthresh):
    fsemb = semblance_power(fptches[iptch,32:])
    dsemb = semblance_power(dptches[iptch,32:])
    rsemb = semblance_power(rptches[iptch,32:])
    dsembrat = dsemb/fsemb
    rsembrat = rsemb/fsemb
    print("Drat=%f Rrat=%f"%(dsembrat,rsembrat))
    if(dsembrat < 0.8):
      print("Defocused")
    if(rsembrat < 0.8):
      print("Residually defocused")
    plot_seglabel(sptches[iptch],lptches[iptch],show=False)
    plot_cubeiso(fptches[iptch,32:],stack=True,elev=15,verb=False,show=False)
    plot_cubeiso(dptches[iptch,32:],stack=True,elev=15,verb=False,show=False)
    plot_cubeiso(rptches[iptch,32:],stack=True,elev=15,verb=False,show=True)


