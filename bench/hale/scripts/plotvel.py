import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

vaxes,vel = sep.read_file("vintzcomb.H")
[nz,nx] = vaxes.n; [oz,ox] = vaxes.o; [dz,dx] = vaxes.d
vel = vel.reshape(vaxes.n,order='F')

fsize = 16
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
im = ax.imshow(vel,cmap='jet',extent=[ox,ox+nx*dx,oz+nz*dz,oz],interpolation='bilinear',aspect=2.0)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.92,0.15,0.02,0.7])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig('./fig/vintz.png',dpi=150,bbox_inches='tight',transparent=True)
#plt.show()

