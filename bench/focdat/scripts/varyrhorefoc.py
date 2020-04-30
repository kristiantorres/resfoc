import inpout.seppy as seppy
import numpy as np
from resfoc.estro import refocusimg
from resfoc.gain import agc
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in images
iaxes,img = sep.read_file("../image/fltimgbigresangstkwrng.H")
img = img.reshape(iaxes.n,order='F')
img = np.ascontiguousarray(img.T).astype('float32')

[nz,nx,nro] = iaxes.n; [dz,dx,dro] = iaxes.d;

# Read in semblance rho
raxes,rsmb = sep.read_file("../image/rhofitnormwrng.H")
rsmb = rsmb.reshape(raxes.n,order='F')
rsmb = np.ascontiguousarray(rsmb.T).astype('float32')

# Read in image space rho
raxes,rimg = sep.read_file("../image/rhotgtimg.H")
rimg = rimg.reshape(raxes.n,order='F')
rimg = np.ascontiguousarray(rimg.T).astype('float32')

# Refocusing
rfis = refocusimg(img,rsmb,dro)

rfii = refocusimg(img,rimg,dro)

# Plotting
vmin = np.min(agc(img[16])); vmax = np.max(agc(img[16]))
pclip = 0.8

plt.figure(1)
plt.imshow(agc(rfis).T,cmap='gray',interpolation='sinc',extent=[0.0,nx*dx/1000.0,(nz)*dz/1000.0,0.0],
           vmin=pclip*vmin,vmax=pclip*vmax)

plt.figure(2)
plt.imshow(agc(rfii).T,cmap='gray',interpolation='sinc',extent=[0.0,nx*dx/1000.0,(nz)*dz/1000.0,0.0],
           vmin=pclip*vmin,vmax=pclip*vmax)

plt.figure(3)
plt.imshow(agc(img[16]).T,cmap='gray',interpolation='sinc',extent=[0.0,nx*dx/1000.0,(nz)*dz/1000.0,0.0],
           vmin=pclip*vmin,vmax=pclip*vmax)

plt.show()
