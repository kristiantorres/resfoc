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
os.environ['CUDA_VISIBLE_DEVICES'] = str(3)

# Read in the sigsbee image
#iaxes,img = sep.read_file("spimgextbobdistr.H")
#iaxes,img = sep.read_file("spimgbobang.H")
#[nz,na,ny,nx] = iaxes.n
#[nz,nx,ny,nhx] = iaxes.n
#img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)
#imgw = img[:,0,:,:]
#imgs = np.sum(imgw,axis=1)
#imgw = img[20,0,:,:]

iaxes,img = sep.read_file("zoimg.H")
[nz,ny,nx] = iaxes.n
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)
imgs = img[:,0,:]

# Size of a single patch
ptchz = 64; ptchx = 64

# Define window
bxw = 20;  exw = nx - 20
bzw = 0; ezq = nz

#imgsw = imgw[bxw:exw,bzw:nz]
imgsw = imgs[bxw:exw,bzw:nz]
#imgg  = agc(imgsw)
imgg  = imgsw
# Transpose
imggt = np.ascontiguousarray(imgg.T)

# Read in the network
with open('./dat/fltsegarch.json','r') as f:
  mdl = model_from_json(f.read())
mdl.load_weights('/scr1/joseph29/hale_fltseg-chkpt.h5')

sigseg,rimg = segmentfaults(imggt,mdl,nzp=ptchz,nxp=ptchx)

print(rimg.shape)
print(sigseg.shape)

vmin = 0.5*np.min(agc(rimg)); vmax = 0.5*np.max(agc(rimg))
plotsegprobs(rimg[0:500],sigseg[0:500],pmin=0.6,show=False,aratio='auto')
plotsegprobs(rimg[0:500],sigseg[0:500],pmin=1.0,show=True,aratio='auto')

