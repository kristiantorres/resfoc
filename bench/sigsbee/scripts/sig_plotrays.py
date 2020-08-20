import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from utils.plot import plot_imgvelptb
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in ray trajectories
raxes,rays = sep.read_file("sig-raysallshots.rsf")
rays = rays.reshape(raxes.n,order='F')
[npts,nrays,nsx] = raxes.n

# Read in velocity model
vaxes,vel = sep.read_file("sigsbee_vel.H")
[nz,nx] = vaxes.n; [dz,dx] = vaxes.d; [oz,ox] = vaxes.o
vel = vel.reshape(vaxes.n,order='F')

# Read in reflecitivity model
iaxes,img = sep.read_file("./dat/ref.H",form='native')
img = img.reshape(iaxes.n,order='F')
imgre = resample(img,[nz,nx])
imin = np.min(img); imax = np.max(img); pclip=0.1

# Read in velocity perturbation
paxes,ptb = sep.read_file("overwdiff.H")
ptb = ptb.reshape(paxes.n,order='F')

# Get coordinates
raysz = np.real(rays).T; raysx = np.imag(rays).T

rayszt = np.transpose(raysz,(0,2,1)); raysxt = np.transpose(raysx,(0,2,1))
jr = 10
raysztj = rayszt[:,:,::jr]; raysxtj = raysxt[:,:,::jr]

# Velocity perturbation
thresh = 5
mask1 = np.ma.masked_where((ptb) <  thresh,ptb)
mask2 = np.ma.masked_where((ptb) > -thresh,ptb)

# Plot velocity on reflectivity
fsize = 16
fig = plt.figure(figsize=(14,7)); ax = fig.gca()
ax.imshow(vel,cmap='jet',interpolation='bilinear',extent=[ox,ox+nx*dx,nz*dz,0])
ax.imshow(img,cmap='gray',interpolation='bilinear',extent=[ox,ox+nx*dx,nz*dz,0],
          vmin=pclip*imin,vmax=pclip*imax,alpha=0.5)
plt.savefig('./fig/sigvel.png',dpi=150,transparent=True,bbox_inches='tight')
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
plt.close()

# Plot perturbation on reflectivity
plot_imgvelptb(imgre,ptb,dz*1000,dx*1000,velmin=-100,velmax=100,thresh=thresh,aagc=False,
               imin=pclip*imin,imax=pclip*imax,figname="./fig/sigvelptb.png",hbar=0.5,barz=0.25)

jsx = 10
for isx in range(0,nsx,jsx):
  fig = plt.figure(figsize=(14,7)); ax = fig.gca()
  ax.imshow(vel,cmap='jet',interpolation='bilinear',extent=[ox,ox+nx*dx,nz*dz,0])
  ax.imshow(img,cmap='gray',interpolation='bilinear',extent=[ox,ox+nx*dx,nz*dz,0],
             vmin=pclip*imin,vmax=pclip*imax,alpha=0.5)
  ax.imshow(mask1,cmap='seismic',interpolation='bilinear',extent=[ox,ox+nx*dx,nz*dz,0],
             vmin=-100,vmax=100,alpha=0.3)
  ax.imshow(mask2,cmap='seismic',interpolation='bilinear',extent=[ox,ox+nx*dx,nz*dz,0],
             vmin=-100,vmax=100,alpha=0.3)
  ax.plot(raysxtj[isx,:,:],raysztj[isx,:,:],color='m')
  ax.set_xlabel('X (km)',fontsize=fsize)
  ax.set_ylabel('Z (km)',fontsize=fsize)
  ax.tick_params(labelsize=fsize)
  plt.savefig('./fig/rays/sigrays%d.png'%(isx),dpi=150,transparent=True,bbox_inches='tight')
  plt.close()
  #plt.show()

