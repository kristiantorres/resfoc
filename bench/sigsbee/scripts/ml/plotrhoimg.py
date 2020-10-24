import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in the image
iaxes,imgn = sep.read_file("stkwind.H")
imgn = imgn.reshape(iaxes.n,order='F')
sc = 0.2
kmin = sc*np.min(imgn); kmax = sc*np.max(imgn)

iaxes,imgp = sep.read_file("stkwindpos.H")
imgp = imgp.reshape(iaxes.n,order='F')
sc = 0.2
kmin = sc*np.min(imgp); kmax = sc*np.max(imgp)

# Read in negative rho prediction
rpaxes,rp = sep.read_file("sigrhopos.H")
rp = rp.reshape(rpaxes.n,order='F')

# Read in positive rho prediction
rnaxes,rn = sep.read_file("sigrho.H")
rn = rn.reshape(rnaxes.n,order='F')

# Rho on image
fsize = 15
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(imgp,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[4.24,25.68,8.763,1.8288],aspect=1.5)
im = ax.imshow(rp,cmap='seismic',interpolation='bilinear',vmin=0.975,vmax=1.025,
               extent=[4.25,25.68,8.736,1.8288],alpha=0.2,aspect=1.5)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.91,0.38,0.02,0.23])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.solids.set(alpha=1)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r'$\rho$',fontsize=fsize)
plt.savefig("./fig/rhoimgpossmb.png",dpi=150,bbox_inches='tight',transparent=True)

fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(imgn,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[4.24,25.68,8.763,1.8288],aspect=1.5)
im = ax.imshow(rn,cmap='seismic',interpolation='bilinear',vmin=0.975,vmax=1.025,
               extent=[4.24,25.68,8.763,1.8288],alpha=0.2,aspect=1.5)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.91,0.38,0.02,0.23])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.solids.set(alpha=1)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r'$\rho$',fontsize=fsize)
plt.savefig("./fig/rhoimgnegsmb.png",dpi=150,bbox_inches='tight',transparent=True)

