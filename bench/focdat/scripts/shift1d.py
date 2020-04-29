import numpy as np
from scipy.ndimage import map_coordinates
from scaas.trismooth import smooth
from utils.movie import viewimgframeskey
import matplotlib.pyplot as plt

# Dimensions
nx = 100; nz = 100; nro = 11
oro = 0.95; dro = 0.01
romax = oro + (nro-1)*dro
rromin = 0.6; rromax = 1.4

rho = np.random.rand(nx,nz)*(rromax-rromin) + rromin
rhosm = smooth(rho.astype('float32'),rect1=20,rect2=20)

img = np.zeros([nro,nx,nz])

for iro in range(nro):
  img[iro,:,:] = oro + iro*dro

ishft = np.zeros([nro,nx,nz])

viewimgframeskey(img,cmap='gray',pclip=1.0)

plt.figure(2)
plt.imshow(rhosm,cmap='seismic',vmin=oro,vmax=romax)
plt.colorbar()

for ix in range(nx):
  for iz in range(nz):
    for iro in range(nro):
      rho = iro*dro + oro
      shiftrho = int(np.round((rhosm[ix,iz]-1.0 + rho-oro)/dro))
      #shift = int(np.round((rhosm[ix,iz] - 1.0)/dro))
      #shiftrho = shift + int((rho-oro)/dro)
      #if(iro + shift < nro and iro + shift >= 0):
      if(shiftrho < nro and shiftrho >= 0):
        ishft[iro,ix,iz] = img[shiftrho,ix,iz]
        #ishft[iro,ix,iz] = img[iro+shift,ix,iz]

plt.figure(3)
plt.imshow(ishft[5],cmap='seismic',vmin=oro,vmax=romax)
plt.colorbar()

coords = np.zeros([3,nro,nx,nz])
for iro in range(nro):
  rho = oro + iro*dro
  for ix in range(nx):
    for iz in range(nz):
      #coords[0,iro,ix,iz] = iro #(rhosm[ix,iz]-1.0 + rho-oro)/dro
      coords[0,iro,ix,iz] = (rhosm[ix,iz]-1.0 + rho-oro)/dro
      coords[1,iro,ix,iz] = ix
      coords[2,iro,ix,iz] = iz

ishft2 = map_coordinates(img,coords,order=1)

plt.figure(4)
plt.imshow(ishft2[5],cmap='seismic',vmin=oro,vmax=romax)
plt.colorbar()
plt.show()

