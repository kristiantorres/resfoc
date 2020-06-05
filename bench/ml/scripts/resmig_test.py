import os
import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import plotsegprobs
from resfoc.resmig import preresmig,get_rho_axis,convert2time
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from deeplearn.keraspredict import segmentfaults
from resfoc.gain import agc
from utils.movie import viewimgframeskey
import matplotlib.pyplot as plt

## Read in the well-focused subsurface offset image
sep = seppy.sep()
#faxes,fog = sep.read_file("../focdat/focdefoc/mltestfog.H")
#fog = fog.reshape(faxes.n,order='F')
#fog = np.ascontiguousarray(fog.T).astype('float32')
#
## Get axes
#[nz,nx,nh] = faxes.n; [oz,ox,oh] = faxes.o; [dz,dx,dh] = faxes.d
#
## Depth Residual migration
#inro = 21; idro = 0.00125
#rmig = preresmig(fog,[dh,dx,dz],nps=[2049,1025,513],nro=inro,dro=idro,time=False,nthreads=18,verb=True)
#onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)
#
## Convert to time
#dt = 0.004
#rmigt = convert2time(rmig,dz,dt=dt,dro=odro,verb=True)
#
## Detect the faults on each residually migrated image
#sep.write_file("fogresmig.H",rmigt.T,ds=[dz,dx,dh,odro],os=[0.0,0.0,oh,ooro])

# Read in the residual migration image
raxes,res = sep.read_file("fogresmig.H")
res = res.reshape(raxes.n,order='F')
[nz,nx,nh,nro] = raxes.n; [oz,ox,oh,oro] = raxes.o; [dz,dx,dh,dro] = raxes.d

res = np.ascontiguousarray(res.T)
reszo = np.transpose(agc(res[:,16,256:768,:]),(0,2,1))

#viewimgframeskey(reszo,transp=False,interp='sinc')

# Set the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
tf.compat.v1.GPUOptions(allow_growth=True)

# Read in the neural network
with open('./dat/networks/xwarch-gpu10.json','r') as f:
  fltmdl = model_from_json(f.read())
fltmdl.load_weights("/net/fantastic/scr2/joseph29/xwarch-gpu10ep-chkpnt.h5")

# Detect faults on each image
fsize = 16
for iro in range(nro):
  rho = iro*dro + oro
  fltprb = segmentfaults(reszo[iro],fltmdl,nzp=128,nxp=128,strdz=64,strdx=64)
  fig = plt.figure(1,figsize=(10,10)); ax = fig.gca()
  ax.imshow(reszo[iro],vmin=-2.5,vmax=2.5,interpolation='sinc',cmap='gray')
  ax.tick_params(labelsize=fsize); ax.set_xlabel(' ',fontsize=fsize); ax.set_ylabel(' ',fontsize=fsize)
  ax.set_title(r'$\rho$=%.3f'%(rho),fontsize=fsize)
  plotsegprobs(reszo[iro],fltprb,pmin=0.3,interp='sinc',vmin=-2.5,vmax=2.5,title=r'$\rho$=%.3f'%(rho),
               wbox=10,hbox=10,barx=0.9,show=True)

