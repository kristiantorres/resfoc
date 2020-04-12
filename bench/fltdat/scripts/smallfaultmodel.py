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
nz = 512; dz = 1.0
nx = 512; dx = 1.0
ny = 256; dy = 10

# Slice in y along which to plot model
slcy = 128

# Model building object
mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,nbase=10,basevel=3000)

nlayer = 200
minvel = 1600; maxvel = 3000
#props = mb.vofz(nlayer,minvel,maxvel,npts=2,persist=0.1,octaves=1)
props = np.linspace(maxvel,minvel,nlayer)
thicks = np.random.randint(5,10,nlayer)

# Loop over all deposits to create the model
csq = 0
for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],layer=100,layer_rand=0.00,dev_layer=0.1)
  #if(ilyr == 2 or ilyr == 20 or ilyr == 27):
  #  amp = np.random.rand()*(100-50) + 50
  #  mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',persist=0.9)

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
cbar_ax = fig.add_axes([0.75,0.3,0.02,0.4])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('velocity (km/s)',fontsize=fsize)
plt.show()

# Faults
azims = [0.0,180.0]

# Tiny faults
#mb.fault(begx=0.2,begz=0.5,begy=0.5,azim=180.0,daz=2000,dz=500,theta_die=9.0,theta_shift=4.0,dist_die=0.3,perp_die=1.0,throwsc=4.0,thresh=20)
#mb.fault(begx=0.35,begz=0.5,begy=0.5,azim=0.0,daz=2000,dz=500,theta_die=9.0,theta_shift=4.0,dist_die=0.3,perp_die=1.0,throwsc=4.0,thresh=20)
mb.fault(begx=0.5,begz=0.1,begy=0.5,azim=180.0,daz=4000,dz=1000,theta_die=9.0,theta_shift=4.0,dist_die=0.6,perp_die=1.0,throwsc=4.0,thresh=20)
#mb.fault(begx=0.65,begz=0.5,begy=0.5,azim=0.0,daz=2000,dz=500,theta_die=9.0,theta_shift=4.0,dist_die=0.3,perp_die=1.0,throwsc=4.0,thresh=20)

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
img = np.array([np.convolve(ref[ix,:],wav) for ix in range(nx)])[:,ns:256+ns]

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

