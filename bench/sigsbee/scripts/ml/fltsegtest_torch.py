import inpout.seppy as seppy
import numpy as np
import torch
from deeplearn.torchnets import Unet
from deeplearn.torchpredict import segmentfaults
from deeplearn.utils import normalize, plotsegprobs
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in the sigsbee image
iaxes,img = sep.read_file("sigsbee_ang.H")
[nz,na,ny,nx] = iaxes.n; [oz,oa,oy,ox] = iaxes.o; [dz,da,dy,dx] = iaxes.d
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)
imgw = img[:,0,:,:]
imgs = np.sum(imgw,axis=1)

# Size of a single patch
ptchz, ptchx = 64, 64
strdz,strdx = ptchz//2, ptchx//2

# Define window
bxw = 20;  exw = nx - 20
bzw = 177; ezw = nz

imgsw = imgs[bxw:exw,bzw:nz]
# Transpose
imggt = np.ascontiguousarray(imgsw.T)

# Read in the torch network
net = Unet()
device = torch.device('cpu')
net.load_state_dict(torch.load('/scr1/joseph29/sigsbee_fltseg_torch_bigagc.pth',map_location=device))

iprb,rimg = segmentfaults(imggt,net,nzp=ptchz,nxp=ptchx)

sc = 0.1
vmin = 0.1*np.min(rimg); vmax = 0.1*np.max(rimg)
plotsegprobs(rimg,iprb,pmin=0.4,show=True,xmin=ox+bxw*dx,xmax=ox+bxw*dx+exw*dx,
             zmin=bzw*dz+oz,zmax=ezw*dz,vmin=vmin,vmax=vmax,fname=None,
             hbar=0.48,barz=0.25,wbox=12,hbox=6,xlabel='X (km)',ylabel='Z (km)')

