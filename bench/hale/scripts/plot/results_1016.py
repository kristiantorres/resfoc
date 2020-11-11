import inpout.seppy as seppy
import numpy as np
from genutils.plot import plot_vel2d
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in the defocused image
daxes,dfc = sep.read_file("faultfocusstk.H")
nz,nx = daxes.n; oz,ox = daxes.o; dz,dx = daxes.d
dfc = dfc.reshape(daxes.n,order='F')
smin = np.min(dfc); smax = np.max(dfc)
pclip = 0.5

# Read in correct velocity
caxes,corr = sep.read_file("vintzcomb.H")
[ovz,ovx] = caxes.o; [dvz,dvx] = caxes.d
corr = corr.reshape(caxes.n,order='F')

# Read in perturbed velocity
paxes,ptb = sep.read_file("velfltfocus1.H")
ptb = ptb.reshape(paxes.n,order='F')

# Read in Rho
raxes,rho = sep.read_file("faultfocusrho.H")
rho = rho.reshape(raxes.n,order='F')

#plot_vel2d(corr,dz=dvz,dx=dvx,ox=ovx,aspect=2.5,figname='./fig/velcorr.png')
#plot_vel2d(ptb,dz=dvz,dx=dvx,ox=ovx,aspect=2.5,figname='./fig/velptb.png')

# Plot rho on top of stack
fsize = 16
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(dfc[:600,10:522],interpolation='bilinear',cmap='gray',vmin=smin*pclip,vmax=smax*pclip,
          extent=[ox,ox+nx*dx,oz+850*dz,oz],aspect=2.5)
im = ax.imshow(rho[:600,10:522],cmap='seismic',interpolation='bilinear',vmin=0.95,vmax=1.05,
               extent=[ox,ox+nx*dx,oz+850*dz,oz],alpha=0.2,aspect=2.5)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.87,0.15,0.02,0.7])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.solids.set(alpha=1)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r'$\rho$',fontsize=fsize)
plt.savefig("./fig/rhofault.png",dpi=150,bbox_inches='tight',transparent=True)

