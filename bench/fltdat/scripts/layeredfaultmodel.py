import numpy as np
import inpout.seppy as seppy
import velocity.mdlbuild as mdlbuild
from scaas.wavelet import ricker
from utils.ptyprint import progressbar
from scaas.trismooth import smooth
import deeplearn.utils as dlut
from scipy.ndimage import gaussian_filter
from utils.signal import bandpass
import matplotlib.pyplot as plt
from matplotlib import colors

# Dimensions of model
nz = 1000;  dz = 12.5
nx = 1000; dx = 25.0
ny = 200;  dy = 25.0

# Slice in y along which to plot model
slcy = 100

# Model building object
mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,nbase=10,basevel=3000)

nlayer = 70
minvel = 1600; maxvel = 3000
props = np.linspace(maxvel,minvel,nlayer)
thicks = np.random.randint(10,20,nlayer)

# Loop over all deposits to create the model
csq = 0
for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],layer=50,layer_rand=0.0,dev_layer=0.1)
  #mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,octaves=1,mode='perlin')
  #mb.deposit(velval=props[ilyr],thick=thicks[ilyr],layer=100,layer_rand=0.0,dev_layer=0.1)

# Water deposit
mb.deposit(1480,thick=10,layer=150,dev_layer=0.0)
# Trim model before faulting
mb.trim(0,nz)
mb.smooth_model(rect1=1,rect2=3,rect3=1)

fsize=18
fig = plt.figure(2,figsize=(14,7))
ax = fig.add_subplot(111)
im = ax.imshow((mb.vel[slcy,:,:]/1000.0).T,cmap='jet',extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0])
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.91,0.3,0.02,0.4])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('velocity (km/s)',fontsize=fsize)
plt.show()

# Fault it up!
mb.fault(begx=0.5,begy=0.5,begz=0.5,daz=8000,dz=5000,azim=0,theta_die=12,theta_shift=4.0,dist_die=0.3,perp_die=0.5,dirf=0.1,throwsc=1.0,thresh=50)
#mb.smallfault(azim=0.0,begz=0.4,begx=0.4,begy=0.5,tscale=10.0)
#mb.smallfault(azim=0.0,begz=0.4,begx=0.5,begy=0.5,tscale=10.0)
#mb.smallfault(azim=0.0,begz=0.4,begx=0.6,begy=0.5,tscale=10.0)

# Plot the faulted model
fig = plt.figure(3,figsize=(10,10))
ax = fig.add_subplot(111)
im = ax.imshow((smooth(mb.vel[slcy,:,:],rect1=1,rect2=2)/1000.0).T,cmap='jet',extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0],
    interpolation='none')
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.show()

vel = mb.vel[slcy,:,:]
lbl = mb.get_label()[slcy,:,:]
ref = mb.get_refl()[slcy,:,:]

nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
amp = 1.0; dly = 0.125; f = 125
# Compute ricker wavelet with random frequency
wav = ricker(nt,dt,f,amp,dly)

# Convolve with reflectivity
img = np.array([np.convolve(ref[ix,:],wav) for ix in range(nx)])[:,ns:nz+ns]

## Plot reflectivity and image with labels
f,axarr = plt.subplots(2,2,figsize=(15,15))
# Reflectivity
axarr[0,0].imshow(ref.T,cmap='gray',extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0])
axarr[0,0].set_ylabel('Z (km)',fontsize=fsize)
axarr[0,0].set_xticks([])
axarr[0,0].tick_params(labelsize=fsize)
# Image
axarr[0,1].imshow(img.T,cmap='gray',extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0])
axarr[0,1].set_yticks([])
axarr[0,1].set_xticks([])
axarr[0,1].tick_params(labelsize=fsize)
# Create mask
mask = np.ma.masked_where(lbl == 0, lbl)
cmap = colors.ListedColormap(['red','white'])
# Reflectivity with label
axarr[1,0].imshow(ref.T,cmap='gray',extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0])
axarr[1,0].imshow(mask.T,cmap,extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0])
axarr[1,0].set_xlabel('X (km)',fontsize=fsize)
axarr[1,0].set_ylabel('Z (km)',fontsize=fsize)
axarr[1,0].tick_params(labelsize=fsize)
# Image with label
axarr[1,1].imshow(img.T,cmap='gray',extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0])
axarr[1,1].imshow(mask.T,cmap,extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0])
axarr[1,1].set_xlabel('X (km)',fontsize=fsize)
axarr[1,1].set_yticks([])
axarr[1,1].tick_params(labelsize=fsize)
plt.subplots_adjust(hspace=0.2)
plt.show()

# Write velocity, reflectivity, image and label to file
sep = seppy.sep()
axes = seppy.axes([nz,nx],[0.0,0.0],[dz,dx])
sep.write_file('velsmall.H',vel.T,ds=[dz,dx])
sep.write_file('refsmall.H',ref.T,ds=[dz,dx])
sep.write_file('lblsmall.H',lbl.T,ds=[dz,dx])
sep.write_file('imgsmall.H',img.T,ds=[dz,dx])

