import numpy as np
import inpout.seppy as seppy
from resfoc.estro import estro_tgt,onehot2rho
from scaas.trismooth import smooth
from utils.movie import viewimgframeskey
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in well-focused image
iaxes,img = sep.read_file("./dat/fltimg-00760.H")
img = img.reshape(iaxes.n,order='F')
izro = img[:,:,16]

# Read in residual migration image
raxes,res = sep.read_file("./dat/resfltimg-00760.H")
res = res.reshape(raxes.n,order='F')
[nz,nx,nro] = raxes.n; [dz,dx,dro] = raxes.d; [oz,ox,oro] = raxes.o

rho,lbls = estro_tgt(res.T,izro.T,dro,oro,strdx=64,strdz=64,onehot=True)

rhob = onehot2rho(lbls,dro,oro)

plt.figure()
plt.imshow(rho.T,extent=[0.0,(nx)*dx,(nz)*dz,0.0],cmap='seismic')
plt.colorbar()
plt.figure()
plt.imshow(smooth(rho.astype('float32'),rect1=100,rect2=100).T,extent=[0.0,(nx)*dx,(nz)*dz,0.0],
    cmap='seismic',vmax=1.02,vmin=0.98)
plt.colorbar()
plt.figure()
plt.imshow(rho.T,extent=[0.0,(nx)*dx,(nz)*dz,0.0],cmap='seismic')
plt.colorbar()

plt.show()

