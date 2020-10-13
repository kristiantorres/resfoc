import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

# Zero offset image
zaxes,zro = sep.read_file("zoimg.H")
zro = zro.reshape(zaxes.n,order='F')
sc = 0.5
zmin = sc*np.min(zro); zmax = sc*np.max(zro)

[nz,ny,nxz] = zaxes.n; [oz,oy,oxz] = zaxes.o; [dz,dy,dxz] = zaxes.d

# Prestack image
iaxes,img = sep.read_file("spimgbob.H")
img = img.reshape(iaxes.n,order='F')
imin = sc*np.min(img); imax = sc*np.max(img)

[nz,ny,nx] = iaxes.n; [oz,oy,ox] = iaxes.o; [dz,dy,dx] = iaxes.d

plt.figure(figsize=(5,10))
plt.imshow(img[:,0,:],cmap='gray',interpolation='sinc',aspect='auto',vmin=imin,vmax=imax)
plt.figure(figsize=(5,10))
plt.imshow(zro[:,0,:],cmap='gray',interpolation='sinc',aspect='auto',vmin=zmin,vmax=zmax)
plt.show()
