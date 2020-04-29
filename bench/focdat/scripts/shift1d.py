import numpy as np
from scipy.ndimage import map_coordinates
from scaas.trismooth import smooth
from resfoc.rhoshifts import rhoshifts
from utils.movie import viewimgframeskey
import matplotlib.pyplot as plt

# Dimensions
nx = 1000; nz = 500; nro = 33; ro1 = int((nro-1)/2)
oro = 0.98; dro = 0.00125
romax = oro + (nro-1)*dro
rromin = 0.9; rromax = 1.1

rho = np.random.rand(nx,nz)*(rromax-rromin) + rromin
rhosm = smooth(rho.astype('float32'),rect1=20,rect2=20)
#rhosm[:,:] = 0.98

img = np.zeros([nro,nx,nz])

for iro in range(nro):
  #print(oro + iro*dro)
  img[iro,:,:] = oro + iro*dro

ishft = np.zeros([nro,nx,nz])

#viewimgframeskey(img,cmap='gray',pclip=1.0)

plt.figure(2)
plt.imshow(rhosm.T,cmap='seismic',vmin=oro,vmax=romax)
plt.colorbar()

#for ix in range(nx):
#  for iz in range(nz):
#    for iro in range(nro):
#      rho = iro*dro + oro
#      shiftrho = int(np.round((rhosm[ix,iz]-1.0 + rho-oro)/dro))
#      #shift = int(np.round((rhosm[ix,iz] - 1.0)/dro))
#      #shiftrho = shift + int((rho-oro)/dro)
#      #if(iro + shift < nro and iro + shift >= 0):
#      if(shiftrho < nro and shiftrho >= 0):
#        ishft[iro,ix,iz] = img[shiftrho,ix,iz]
#        #print(iro,iro+shift)
#        #ishft[iro,ix,iz] = img[iro+shift,ix,iz]

#plt.figure(3)
#plt.imshow(ishft[ro1],cmap='seismic',vmin=oro,vmax=romax)
#plt.colorbar()
#plt.show()

coords = np.zeros([3,nro,nx,nz],dtype='float32')

rhoshifts(nro,nx,nz,dro,rhosm,coords)
#for iro in range(nro):
#  rho = oro + iro*dro
#  for ix in range(nx):
#    for iz in range(nz):
#      #coords[0,iro,ix,iz] = (rhosm[ix,iz]-1.0 + rho-oro)/dro
#      coords[0,iro,ix,iz] = (rhosm[ix,iz]-1.0)/dro + iro
#      coords[1,iro,ix,iz] = ix
#      coords[2,iro,ix,iz] = iz

ishft2 = map_coordinates(img,coords)

plt.figure(4)
plt.imshow(ishft2[ro1].T,cmap='seismic',vmin=oro,vmax=romax)
plt.colorbar()
plt.show()

