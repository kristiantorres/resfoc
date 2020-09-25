import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in the image
iaxes,img = sep.read_file("stkfocwind.H")
img = img.reshape(iaxes.n,order='F')
sc = 0.2
kmin = sc*np.min(img); kmax = sc*np.max(img)

# Read in negative rho prediction
rpaxes,rp = sep.read_file("sigfocrhopos.H")
rp = rp.reshape(rpaxes.n,order='F')

# Read in positive rho prediction
rnaxes,rn = sep.read_file("sigfocrho.H")
rn = rn.reshape(rnaxes.n,order='F')

# Rho on image
fsize = 15
#fig = plt.figure(figsize=(10,10)); ax = fig.gca()
#ax.imshow(img,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
#          extent=[5,25,8,2])
#im = ax.imshow(rp,cmap='seismic',interpolation='bilinear',vmin=0.975,vmax=1.025,
#               extent=[5,25,8,2],alpha=0.2)
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#cbar_ax = fig.add_axes([0.91,0.38,0.02,0.23])
#cbar = fig.colorbar(im,cbar_ax,format='%.2f')
#cbar.solids.set(alpha=1)
#cbar.ax.tick_params(labelsize=fsize)
#cbar.set_label(r'$\rho$',fontsize=fsize)
#plt.savefig("./fig/rhoimgposcnn.png",dpi=150,bbox_inches='tight',transparent=True)

fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(img,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[5,25,8,2])
im = ax.imshow(rn,cmap='seismic',interpolation='bilinear',vmin=0.975,vmax=1.025,
               extent=[5,25,8,2],alpha=0.2)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.91,0.38,0.02,0.23])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.solids.set(alpha=1)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r'$\rho$',fontsize=fsize)
plt.savefig("./fig/rhoimgnegcnn.png",dpi=150,bbox_inches='tight',transparent=True)

