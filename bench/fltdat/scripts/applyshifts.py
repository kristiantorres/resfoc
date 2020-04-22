import inpout.seppy as seppy
from scaas.trismooth import smooth
from scipy.ndimage import map_coordinates
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

zaxes,shiftz = sep.read_file('shiftz.H')
shiftz = shiftz.reshape(zaxes.n,order='F').T
xaxes,shiftx = sep.read_file('shiftx.H')
shiftx = shiftx.reshape(xaxes.n,order='F').T
laxes,lyrs = sep.read_file('layers.H')
lyrs = lyrs.reshape(laxes.n,order='F').T
velflt = np.zeros(lyrs.shape).T

[nz,nx] = laxes.n; [dz,dx] = laxes.d

## Apply shifts
#for ix in range(nx):
#  x = ix*dx
#  for iz in range(nz):
#    z = iz*dz
#    l1 = max(0,int(((z - shiftz[ix,iz])/dz + 0.5)))
#    l2 = max(0,int(((x - shiftx[ix,iz])/dx + 0.5)))
#    if(l1 >= nz): l1 = nz - 1 
#    if(l2 >= nx): l2 = nx - 1 
#    if(l1 >= 0): 
#      # From desination to source
#      velflt[ix,iz] = lyrs[l2,l1]
#      # From source to destination
#      #velflt[l2,l1] = lyrs[ix,iz]
#
#plt.imshow(velflt.T,cmap='jet')
#plt.show()

szs = np.zeros([nx,nz])
sxs = np.zeros([nx,nz])

gx=np.linspace(0,nx-1,nx)
gz=np.linspace(0,nz-1,nz)

coords = np.zeros([2,1000,1000])

for ix in range(nx):
  x = ix*dx
  for iz in range(nz):
    z = iz*dz
    sz = (z - shiftz[ix,iz])/dz
    szs[ix,iz] = sz
    sx = (x - shiftx[ix,iz])/dx
    sxs[ix,iz] = sx
    coords[0,ix,iz] = sx
    coords[1,ix,iz] = sz
    #floorz = int(np.floor(sz)); ceilz = int(np.ceil(sz))
    #floorx = int(np.floor(sx)); ceilx = int(np.ceil(sx))
    #deltax = sx - floorx
    #deltaz = sz - floorz
    ## Interpolate between the corners
    #topleft = lyrs[floorx,floorz]
    #toprite = lyrs[floorx,ceilz]
    #botleft = lyrs[ceilx,floorz]
    #botrite = lyrs[ceilx,ceilz]
    #top = (1-deltax) * topleft + deltax * toprite
    #bot = (1-deltax) * botleft + deltax * botrite
    #velflt[ix,iz] = top * (1-deltaz) + deltaz * bot

#plt.imshow(smooth(velflt.astype('float32'),rect1=3,rect2=3).T,cmap='jet')
#plt.show()

fltnew = map_coordinates(lyrs,coords)

plt.imshow(fltnew.T,cmap='jet')
plt.show()

