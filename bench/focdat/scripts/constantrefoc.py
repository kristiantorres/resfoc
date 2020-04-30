import inpout.seppy as seppy
import numpy as np
from resfoc.estro import refocusimg
import matplotlib.pyplot as plt

sep = seppy.sep()

iaxes,img = sep.read_file("../image/fltimgbigresangstkshift.H")
img = img.reshape(iaxes.n,order='F')
img = np.ascontiguousarray(img.T).astype('float32')

[nz,nx,nro] = iaxes.n; [dz,dx,dro] = iaxes.d

# Create constant rho map
rho = np.zeros([nx,nz],dtype='float32')

rho[:] = 0.9925

rfi = refocusimg(img,rho,dro)

plt.figure(1)
plt.imshow(rfi.T,cmap='gray',interpolation='sinc',extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0])

plt.figure(2)
plt.imshow(img[16,:,:].T,cmap='gray',interpolation='sinc',extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0])

plt.show()
