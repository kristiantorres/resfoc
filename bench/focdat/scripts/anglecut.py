import inpout.seppy as seppy
from resfoc.estro import anglemask
from scaas.noise_generator import perlin
from scaas.trismooth import smooth
import numpy as np
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey

sep = seppy.sep()
aaxes,ang = sep.read_file('./dat/refocus/mltest/mltestdogang2.H')
ang = ang.reshape(aaxes.n,order='F').T

# Get dimensions
[nz,na,nx,nro] = aaxes.n; [oz,oa,ox,oro] = aaxes.o; [dz,da,dx,dro] = aaxes.d

# Create the mask
mask = anglemask(nz,na,zpos=0.05,apos=0.6,mode='slant',rand=True,rectz=10,recta=10).T

sep.write_file('mask.H',mask.T,os=[0,oa],ds=[dz,da])

plt.figure(1)
plt.imshow(mask.T,cmap='gray')
plt.show()

# Replicate along the spatial dimension
maskrep = np.repeat(mask[np.newaxis,:,:],nx,axis=0)

# Apply the mask to each rho
masked = np.asarray([maskrep*ang[iro] for iro in range(nro)])

# Add small random noise
masked += (np.random.rand(*masked.shape)*2-1)/10000.0

# Write the masked data to file
sep.write_file('./dat/refocus/mltest/mltestdogang3mask.H',masked.T,ds=aaxes.d,os=aaxes.o)
