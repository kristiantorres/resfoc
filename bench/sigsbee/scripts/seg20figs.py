import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import patch_window2d
from scaas.velocity import salt_mask
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from genutils.plot import plot_imgvelptb

sep = seppy.sep()
aaxes,ang = sep.read_file("sigsbee_ang.H")
ang  = ang.reshape(aaxes.n,order='F').T
[nz,na,ny,nx] = aaxes.n; [dz,da,dy,dx] = aaxes.d; [oz,oa,oy,ox] = aaxes.o
# Window in x and in angle
bxw = 20; nxw = nx - 20
angw = ang[bxw:nxw,0,32:,:]
stk = np.sum(angw,axis=1)
sc = 0.1
smin = sc*np.min(stk); smax = sc*np.max(stk)
#
## Region of interest
oxw = ox + bxw*dx
xmin = 50; xmax = 200
zmin = 160; zmax=1100

# Plot the velocity error on the image
eaxes,err = sep.read_file("overwinterp.H")
err = err.reshape(eaxes.n,order='F').T
errw = -1*err[bxw:nxw,:]

# Plot perturbation on reflectivity
#ax = plot_imgvelptb(stk.T,errw.T,dz,dx,velmin=-100,velmax=100,thresh=5,aagc=False,
#                    imin=smin,imax=smax,figname='./fig/seg20figs/sigvelptb.png',hbar=0.53,barz=0.23,
#                    xmin=oxw,xmax=(oxw+nxw*dx),close=False)

#rect = patches.Rectangle((6.53,3.048),6.86,5.334,linewidth=2,edgecolor='yellow',facecolor='none')

ax.add_patch(rect)

plt.savefig("./fig/seg20figs/sigvelptbbox.png",transparent=True,dpi=150,bbox_inches='tight')
plt.close()

# Plot the regions of interest
seaxes,sestk = sep.read_file("stkfocwindpostorch.H")
sestk = np.ascontiguousarray(sestk.reshape(seaxes.n,order='F'))
sc2 = 0.2
kmin = sc2*np.min(sestk); kmax = sc2*np.max(sestk)

# Region of interest
xmin2 = 50; xmax2 = 200
zmin2 = 160; zmax2=1100

fsize = 15
fig = plt.figure(figsize=(12,6)); ax = fig.gca()
ax.imshow(sestk[zmin2:zmax2,xmin2:xmax2],interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[oxw+xmin*dx,oxw+xmax*dx,zmax*dz,zmin*dz])
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.savefig("./fig/seg20figs/stkwind.png",dpi=150,bbox_inches='tight',transparent=True)

# Read in the velocity model
vaxes,vel = sep.read_file("sigoverw_velint.H")
vel = vel.reshape(vaxes.n,order='F').T
velw = vel[bxw:nxw,0,:]

bzw = 240; ezw = 1150

velww = velw[:,bzw:ezw].T
stkww = stk[:,bzw:ezw].T

print(velww.shape,stkww.shape)
velb = patch_window2d(velww,64,64)
stkb = patch_window2d(stkww,64,64)
print(velb.shape,stkb.shape)

# Apply a mask to the well focused image
msk,stkwbmsk = salt_mask(stkb,velb,saltvel=4.5)

# Window it based on the patch grids

# Plot
fig = plt.figure(figsize=(12,6)); ax = fig.gca()
ax.imshow(stkwbmsk[zmin2:zmax2,xmin2:xmax2],interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[oxw+xmin*dx,oxw+xmax*dx,zmax*dz,zmin*dz])
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.savefig("./fig/seg20figs/focuswind.png",dpi=150,bbox_inches='tight',transparent=True)

raxes,rfi = sep.read_file("sigfocrfipostorch.H")
rfi = np.ascontiguousarray(rfi.reshape(raxes.n,order='F'))

fig = plt.figure(figsize=(12,6)); ax = fig.gca()
ax.imshow(rfi[zmin2:zmax2,xmin2:xmax2],interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[oxw+xmin*dx,oxw+xmax*dx,zmax*dz,zmin*dz])
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.savefig("./fig/seg20figs/rfiwind.png",dpi=150,bbox_inches='tight',transparent=True)


