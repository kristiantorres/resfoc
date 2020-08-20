import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in stack
saxes,stk = sep.read_file("halestk.H")
stk = stk.reshape(saxes.n,order='F')
[oz,ox] = saxes.o; [nz,nx] = saxes.n; [dz,dx] = saxes.d
pclip = 0.8
smin = np.min(stk); smax = np.max(stk)

# Read in rho
raxes,rho = sep.read_file("halerho.H")
rho = rho.reshape(raxes.n,order='F')

# Plot rho on top of stack
fsize = 16
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(stk[:850,:],interpolation='bilinear',cmap='gray',vmin=smin*pclip,vmax=smax*pclip,
          extent=[ox,ox+nx*dx,oz+850*dz,oz],aspect=2.5)
im = ax.imshow(rho[:850,:],cmap='seismic',interpolation='bilinear',vmin=0.95,vmax=1.05,
               extent=[ox,ox+nx*dx,oz+850*dz,oz],alpha=0.2,aspect=2.5)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.87,0.15,0.02,0.7])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.solids.set(alpha=1)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r'$\rho$',fontsize=fsize)
plt.savefig("./fig/rhoimg.png",dpi=150,bbox_inches='tight',transparent=True)
#plt.show()

