import numpy as np
from scaas.trismooth import smooth
from resfoc.estro import refocusimg
import matplotlib.pyplot as plt 

# Dimensions
nx = 1000; nz = 500; nro = 33; ro1 = int((nro-1)/2)
oro = 0.98; dro = 0.00125
romax = oro + (nro-1)*dro
rromin = 0.9; rromax = 1.1 

rho = np.random.rand(nx,nz)*(rromax-rromin) + rromin
rhosm = smooth(rho.astype('float32'),rect1=20,rect2=20)

img = np.zeros([nro,nx,nz])

for iro in range(nro):
  img[iro,:,:] = oro + iro*dro

refi = refocusimg(img,rhosm,dro)

plt.figure(1)
plt.imshow(rhosm.T,cmap='seismic',vmin=oro,vmax=romax)
plt.colorbar()

plt.figure(2)
plt.imshow(refi.T,cmap='seismic',vmin=oro,vmax=romax)
plt.colorbar()

plt.show()

