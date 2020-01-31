import numpy as np
import inpout.seppy as seppy
import matplotlib.pyplot as plt

# Set up io
sep = seppy.sep([])

fsize=18
## Correct image
# Read in the image
eiaxes,eimg = sep.read_file(None,ifname='eimgttpow.H')
nz = eiaxes.n[0]; dz = eiaxes.d[0]
nx = eiaxes.n[1]; dx = eiaxes.d[1]
nh = eiaxes.n[2]; oh = eiaxes.o[2]; dh = eiaxes.d[2]
eimg = eimg.reshape(eiaxes.n,order='F')
imgpt = 176
# Make the plot
f1,ax1 = plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios': [6, 1]})
ax1[0].imshow(eimg[:,:,10],extent=[0,(nx-1)*dx/1000.0,(nz-1)*dz/1000.0,0],cmap='gray')
lz = np.linspace(0.0,(nz-1)*dz/1000.0,nz)
lx = np.zeros(nz) + imgpt*dx/1000.0
ax1[0].plot(lx,lz,color='k',linewidth=2)
ax1[0].set_xlabel('x (km)',fontsize=fsize)
ax1[0].set_ylabel('z (km)',fontsize=fsize)
ax1[0].tick_params(labelsize=fsize)
ax1[1].imshow(eimg[:,imgpt,:],extent=[oh/1000.0,-oh/1000.0,(nz-1)*dz/1000.0,0],cmap='gray',aspect=0.3)
ax1[1].set_xlabel('h (km)',fontsize=fsize)
ax1[1].set_yticks([])
ax1[1].tick_params(labelsize=fsize)
plt.subplots_adjust(wspace=0.2)
plt.savefig('./report/fall2019/fig/trueimg.png',bbox_inches='tight',dpi=150)


# Wrongly migrated image
wiaxes,wimg = sep.read_file(None,ifname='ewrngimgttpow.H')
nz = wiaxes.n[0]; dz = wiaxes.d[0]
nx = wiaxes.n[1]; dx = wiaxes.d[1]
nh = wiaxes.n[2]; oh = wiaxes.o[2]; dh = wiaxes.d[2]
wimg = wimg.reshape(wiaxes.n,order='F')
imgpt = 176
# Make the plot
f2,ax2 = plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios': [6, 1]})
ax2[0].imshow(wimg[:,:,10],extent=[0,(nx-1)*dx/1000.0,(nz-1)*dz/1000.0,0],cmap='gray')
lz = np.linspace(0.0,(nz-1)*dz/1000.0,nz)
lx = np.zeros(nz) + imgpt*dx/1000.0
ax2[0].plot(lx,lz,color='k',linewidth=2)
ax2[0].set_xlabel('x (km)',fontsize=fsize)
ax2[0].set_ylabel('z (km)',fontsize=fsize)
ax2[0].tick_params(labelsize=fsize)
ax2[1].imshow(wimg[:,imgpt,:],extent=[oh/1000.0,-oh/1000.0,(nz-1)*dz/1000.0,0],cmap='gray',aspect=0.3)
ax2[1].set_xlabel('h (km)',fontsize=fsize)
ax2[1].set_yticks([])
ax2[1].tick_params(labelsize=fsize)
plt.subplots_adjust(wspace=0.2)
plt.savefig('./report/fall2019/fig/wrongimg.png',bbox_inches='tight',dpi=150)

# The rho map
raxes,rho = sep.read_file(None,ifname='rrho.H')
rho = rho.reshape(raxes.n,order='F')
f3,ax3 = plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios': [6, 1]})
im3 = ax3[0].imshow(rho,extent=[0,(nx-1)*dx/1000.0,(nz-1)*dz/1000.0,0],cmap='jet')
ax3[0].set_xlabel('x (km)',fontsize=fsize)
ax3[0].set_ylabel('z (km)',fontsize=fsize)
ax3[0].tick_params(labelsize=fsize)
ax3[1].axis('off')
cbar_ax = f3.add_axes([0.75,0.21,0.02,0.57])
cbar = f3.colorbar(im3,cbar_ax,format='%.2f')
cbar.set_label(r'$\rho$',fontsize=fsize)
cbar.ax.tick_params(labelsize=fsize)
cbar.draw_all()
plt.savefig('./report/fall2019/fig/rhoimg.png',bbox_inches='tight',dpi=150)

plt.show()
