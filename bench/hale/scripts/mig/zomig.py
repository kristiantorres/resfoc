import inpout.seppy as seppy
import numpy as np
import oway.zerooffset as zo
from oway.costaper import costaper
import matplotlib.pyplot as plt

# IO
sep = seppy.sep()

# Read in data
daxes,dat = sep.read_file("stk.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
dat = np.expand_dims(dat,axis=0)
[nt,nxi] = daxes.n; [ot,oxi] = daxes.o; [dt,dxi] = daxes.d

# Read in the velocity model
vaxes,vel = sep.read_file("vintz.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nvx],dtype='float32')
velin[:,0,:] = vel

print(dat.shape,velin.shape)

# Normalize the data by the max
datn = dat/np.max(dat)
# Apply a taper to the data at the edges
datt = costaper(datn,nw2=10)

# Plot the velocity model and zero-offset data
dmin = np.min(datt); dmax = np.max(datt); sc = 0.1
fig = plt.figure(figsize=(14,7)); ax = fig.gca()
ax.imshow(datt[0].T,cmap='gray',interpolation='sinc',extent=[oxi,oxi+(nxi)*dxi,nt*dt,0],
         vmin=dmin*sc,vmax=dmax*sc)
fig = plt.figure(figsize=(14,7)); ax = fig.gca()
ax.imshow(velin[:,0,:],cmap='jet',interpolation='bilinear',extent=[ovx,ovx+nvx*dvx,nz*dz,0])
plt.show()

zomig = zo.zerooffset(nxi,dxi,ny,dy,nz,dz,ox=oxi)

img = zomig.image_data(datt,dt,ntx=4,minf=0,maxf=50.1,jf=1,vel=velin,nrmax=10,nthrds=4)

# Plot image
imin = np.min(img); imax = np.max(img); sc=1.0
fig = plt.figure(figsize=(14,7)); ax = fig.gca()
ax.imshow(img[:,0,:],cmap='gray',interpolation='sinc',extent=[oxi,oxi+(nxi)*dxi,nz*dz,0],
         vmin=imin*sc,vmax=imax*sc)
plt.show()

sep.write_file("zoimg.H",img,ds=[dz,dy,dxi],os=[0,0,oxi])

