import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
import subprocess
import torch
from deeplearn.torchnets import Unet
from deeplearn.torchpredict import segmentfaults
from deeplearn.utils import plot_segprobs, normextract
from genutils.plot import plot_img2d

# Read in the image
sep = seppy.sep()
#iaxes,img = sep.read_file("spimgbobang.H")
iaxes,img = sep.read_file("spimgbobangwrng.H")
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)[:,0,:,:]
stkw = agc(np.sum(img,axis=1))[30:542,100:356]
dz,da,dy,dx = iaxes.d; oz,oa,oy,ox = iaxes.o

iaxes,img = sep.read_file("faultfocusang.H")
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)[:,:,:]
stkw = agc(np.sum(img,axis=1))[10:522,100:356]

# Perform structure-oriented smoothing
smooth = True
if(smooth):
  sep.write_file("presmooth.H",stkw.T)
  sp = subprocess.check_call("python scripts/SOSmoothing.py -fin presmooth.H -fout smooth.H",shell=True)
  saxes,smt = sep.read_file("smooth.H")
  smt = np.ascontiguousarray(smt.reshape(saxes.n,order='F'))
else:
  smt = np.ascontiguousarray(stkw.T)

# Read in the torch network
net = Unet()
device = torch.device('cpu')
net.load_state_dict(torch.load('/scr1/joseph29/hale2_fltsegsm.pth',map_location=device))
#net.load_state_dict(torch.load('/scr1/joseph29/hale2_fltsegnosm.pth',map_location=device))

# Segment the faults
ptchz,ptchx = 128,128
iprb = segmentfaults(smt,net,nzp=ptchz,nxp=ptchx)

# Plot the prediction
#plot_img2d(smt,pclip=0.5,show=False)
plot_segprobs(smt,iprb,pmin=0.2,oz=100*dz,dz=dz,ox=30*dx+ox,dx=dx,
              show=False,pclip=0.5,aspect=3.0,labelsize=14,ticksize=14,barlabelsize=14,
              hbar=0.45,barz=0.27,cropsize=140,fname='./fig/segfigs/smoothres')
plot_segprobs(stkw.T,iprb,pmin=0.2,oz=100*dz,dz=dz,ox=30*dx+ox,dx=dx,
              show=False,pclip=0.5,aspect=3.0,labelsize=14,ticksize=14,barlabelsize=14,
              hbar=0.45,barz=0.27,cropsize=140,fname='./fig/segfigs/noisyres')

