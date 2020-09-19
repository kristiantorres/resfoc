import os
import inpout.seppy as seppy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from resfoc.gain import agc
from deeplearn.keraspredict import segmentfaults
from deeplearn.utils import plotsegprobs
import matplotlib.pyplot as plt

sep = seppy.sep()
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

# Read in the sigsbee image
iaxes,img = sep.read_file("sigsbee_ang.H")
[nz,na,ny,nx] = iaxes.n; [oz,oa,oy,ox] = iaxes.o; [dz,da,dy,dx] = iaxes.d
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)
imgw = img[:,0,:,:]
imgs = np.sum(imgw,axis=1)

# Size of a single patch
ptchz = 64; ptchx = 64

# Define window
bxw = 20;  exw = 480
#bzw = 177; ezq = nz
bzw = 240; ezw = 1150

imgsw = imgs[bxw:exw,bzw:nz]
#imgg  = agc(imgsw)
imgg  = imgsw
# Transpose
imggt = np.ascontiguousarray(imgg.T)

# Read in the network
with open('./dat/fltsegarch.json','r') as f:
  mdl = model_from_json(f.read())
mdl.load_weights('/scr1/joseph29/sigsbee_fltseg-chkpt.h5')

sigseg,rimg = segmentfaults(imggt,mdl,nzp=ptchz,nxp=ptchx)

print(rimg.shape)
print(sigseg.shape)

sc = 0.1
vmin = 0.1*np.min(rimg); vmax = 0.1*np.max(rimg)
plotsegprobs(rimg,sigseg,pmin=0.4,show=False,xmin=ox+bxw*dx,xmax=ox+bxw*dx+exw*dx,
             zmin=bzw*dz+oz,zmax=ezw*dz,vmin=vmin,vmax=vmax,fname='./fig/semfigs/sigsbeeseg',
             hbar=0.48,barz=0.25,wbox=12,hbox=6,xlabel='X (km)',ylabel='Z (km)',
             cropsize=180)

