import os
import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import plotsegprobs
from resfoc.resmig import preresmig,get_rho_axis,convert2time
from resfoc.estro import refocusimg
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from deeplearn.keraspredict import segmentfaults
from resfoc.gain import agc
from genutils.movie import viewimgframeskey
import matplotlib.pyplot as plt

## Read in the well-focused subsurface offset image
sep = seppy.sep()
# Read in the residual migration image
raxes,res = sep.read_file("fogresmig.H")
res = np.ascontiguousarray(res.reshape(raxes.n,order='F').T)
[nz,nx,nh,nro] = raxes.n; [oz,ox,oh,oro] = raxes.o; [dz,dx,dh,dro] = raxes.d

# Get the well focused image
resfoc = agc(res[20,16,256:768,:]).T

# Create stationarily defocused image
ros = np.linspace(oro,oro + (nro-1)*dro,nro)

nnro = 21; noro = 0.9875; dro = 0.00125
nros = np.linspace(noro,noro + (nnro-1)*dro,nnro)

# Write the constant shifted images out
fshift = 5
resw = res[fshift:fshift+nnro,:,:,:]
print(ros[fshift:fshift+nnro])
#sep.write_file("foimgresshift.H",resw.T,ds=[dz,dx,dh,dro],os=[0.0,0.0,oh,noro])

reszo = np.transpose(agc(resw[:,16,256:768,:]),(0,2,1))

nro,nxw,nzw = reszo.shape
rhoshift = np.zeros([nxw,nzw],dtype='float32')
rhoshift[:] = 1.00625

# Shift the rho=1 image so it appears defocused

viewimgframeskey(reszo,transp=False,interp='sinc',ottl=noro,dttl=dro,ttlstring=r'$\rho$=%.5f')

rfishift = refocusimg(reszo,rhoshift,dro)

# Set the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
tf.compat.v1.GPUOptions(allow_growth=True)

# Read in the neural network
with open('./dat/networks/xwarch-gpu10.json','r') as f:
  fltmdl = model_from_json(f.read())
fltmdl.load_weights("/net/fantastic/scr2/joseph29/xwarch-gpu10ep-chkpnt.h5")

# Detect faults on refocused image
fsize = 16
fltprb = segmentfaults(rfishift,fltmdl,nzp=128,nxp=128,strdz=64,strdx=64)
fig = plt.figure(1,figsize=(10,10)); ax = fig.gca()
ax.imshow(rfishift,vmin=-2.5,vmax=2.5,interpolation='sinc',cmap='gray')
ax.tick_params(labelsize=fsize); ax.set_xlabel(' ',fontsize=fsize); ax.set_ylabel(' ',fontsize=fsize)
ax.set_title('Refocused',fontsize=fsize)
plotsegprobs(rfishift,fltprb,pmin=0.3,interp='sinc',vmin=-2.5,vmax=2.5,
             wbox=10,hbox=10,barx=0.9,show=True)

prbfoc = segmentfaults(resfoc,fltmdl,nzp=128,nxp=128,strdz=64,strdx=64)
fig = plt.figure(2,figsize=(10,10)); ax = fig.gca()
ax.imshow(resfoc,vmin=-2.5,vmax=2.5,interpolation='sinc',cmap='gray')
ax.tick_params(labelsize=fsize); ax.set_xlabel(' ',fontsize=fsize); ax.set_ylabel(' ',fontsize=fsize)
ax.set_title('Well focused',fontsize=fsize)
plotsegprobs(resfoc,fltprb,pmin=0.3,interp='sinc',vmin=-2.5,vmax=2.5,
             wbox=10,hbox=10,barx=0.9,show=True)

