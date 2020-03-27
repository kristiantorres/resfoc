#!/usr/bin/env python
# coding: utf-8

# # Creating synthetic models for fault classification
# 
# This notebook illustrates how to use the software in the velocity directory of this repo to create synthetic models for fault classification

# ## Define parameters of model
# 
# I first define the parameters of the model that will be created using the model building program. With the defined parameters, I can create the model building object that will be used to create each feature of the model

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import velocity.mdlbuild as mdlbuild
from scaas.wavelet import ricker
from utils.ptyprint import progressbar
import utils.rand as rndut

# Dimensions of the model
nz = 1000; dz = 12.0
nx = 1000; dx = 25.0
ny = 200;  dy = 25.0
# Slice in y along which to plot model
slcy = 100

# Model building object
mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)


# Next I define the number of layers and the parameters for the velocity gradient as well as the thicknesses of the layers

# In[2]:


nlayer = 20
minvel = 1600; maxvel = 5000
props = mb.vofz(nlayer,minvel,maxvel)
thicks = np.random.randint(40,61,nlayer)

# Plot the v(z) gradient
fsize = 18
fig = plt.figure(1,figsize=(6,6))
ax = fig.add_subplot(111)
dlayer = ((nz-1)*dz)/nlayer
z = np.linspace(0.0,(nlayer-1)*dlayer,nlayer)
ax.plot(np.flip(props/1000.0),z/1000.0)
ax.set_xlabel('Velocity (km/s)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
ax.invert_yaxis()
plt.savefig('./fig/voz.png',bbox_inches='tight',dpi=150,transparent=True)
plt.close()


# Next I determine at which point of the deposition I desire to fold the layers. The first argument controls the number of layers to fold and the last determined the minimum spacing between the layers to be folded

# In[3]:


sqlyrs = sorted(mb.findsqlyrs(3,nlayer,5))
print("Folding lyrs %d, %d and %d"%(sqlyrs[0],sqlyrs[1],sqlyrs[2]))


# ## Create a folded model
# I first create a midly-folded model that has the velocity gradient shown in the figure above. To do this I use the deposit and squish functions within model build.

# In[4]:


# Loop over all deposits to create the model
csq = 0
for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],layer=150,layer_rand=0.00,dev_layer=0.1)
  # Random folding
  if(ilyr in sqlyrs):
    if(sqlyrs[csq] < 15):
      # Random amplitude variation in the folding
      amp = np.random.rand()*(3000-500) + 500 
      mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
    elif(sqlyrs[csq] >= 15 and sqlyrs[csq] < 18):
      amp = np.random.rand()*(1800-500) + 500 
      mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
    else:
      amp = np.random.rand()*(500-300) + 300
      mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
    csq += 1

# Water deposit
mb.deposit(1480,thick=50,layer=150,dev_layer=0.0)
# Trim model before faulting
mb.trim(0,nz+100)

fsize = 20
fig = plt.figure(2,figsize=(14,7)); ax = fig.gca()
im = ax.imshow((mb.vel[slcy,:,:]/1000.0).T,cmap='jet',extent=[0,(nx-1)*10/1000.0,(nz+100-1)*5/1000.0,0.0],vmin=1.5,vmax=4.5)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.91,0.11,0.02,0.77])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('velocity (km/s)',fontsize=fsize)
plt.savefig('./fig/velnofaults.png',bbox_inches='tight',dpi=150,transparent=True)
plt.close()

# ## Adding faults to the folded model
# I now add many faults to this folded model. I add larger faults at the bottom of the model, and gradually increase the fault size with depth. The relative position of the fault will in the end not matter for the neural network, but it can affect the imaging.

# In[5]:


# Fault it up!
azims = [0.0,180.0]

# Large faults
nlf = np.random.randint(2,5)
for ifl in progressbar(range(nlf), "nlfaults:", 40):
  azim = np.random.choice(azims)
  xpos = rndut.randfloat(0.1,0.9)
  mb.largefault(azim=azim,begz=0.65,begx=xpos,begy=0.5,tscale=6.0)

# Medium faults
nmf = np.random.randint(3,6)
for ifl in progressbar(range(nmf), "nmfaults:", 40):
  azim = np.random.choice(azims)
  xpos = rndut.randfloat(0.05,0.95)
  mb.mediumfault(azim=azim,begz=0.65,begx=xpos,begy=0.5,tscale=3.0)

# Small faults (sliding or small)
nsf = np.random.randint(5,10)
for ifl in progressbar(range(nsf), "nsfaults:", 40):
  azim = np.random.choice(azims)
  xpos = rndut.randfloat(0.05,0.95)
  zpos = rndut.randfloat(0.2,0.5)
  mb.smallfault(azim=azim,begz=zpos,begx=xpos,begy=0.5,tscale=2.0)

# Tiny faults
ntf = np.random.randint(5,10)
for ifl in progressbar(range(ntf), "ntfaults:", 40):
  azim = np.random.choice(azims)
  xpos = rndut.randfloat(0.05,0.95)
  zpos = rndut.randfloat(0.15,0.3)
  mb.tinyfault(azim=azim,begz=zpos,begx=xpos,begy=0.5,tscale=2.0)

# Plot the faulted model
fig = plt.figure(3,figsize=(14,7)); ax = fig.gca()
im = ax.imshow((mb.vel[slcy,:,:]/1000.0).T,cmap='jet',extent=[0,(nx-1)*10/1000.0,(nz+100-1)*5/1000.0,0.0],vmin=1.5,vmax=4.5)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.91,0.11,0.02,0.77])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('velocity (km/s)',fontsize=fsize)
plt.savefig('./fig/velfaults.png',bbox_inches='tight',dpi=150,transparent=True)
plt.close()


# ## Model processing
# Now with the created velocity model, I can reshape the model to the output I want (512 x 1024) and also compute the reflectivity and a zero-offset image by taking a vertical derivative and then convolving with a wavelet.

# In[6]:


import deeplearn.utils as dlut
from scipy.ndimage import gaussian_filter
from utils.signal import bandpass

## Resize the outputs
nzo = 512; nxo = 1024
# Smooth the model slightly and interpolate to the desired output
velsm = gaussian_filter(mb.vel[slcy,:,:nz],sigma=0.5)
velo = dlut.resample(velsm,[nxo,nzo],kind='linear').T
# Get the labels from the model builder object
lblo = dlut.thresh(dlut.resample(mb.get_label()[slcy,:,:nz],[nxo,nzo],kind='linear'),0.0).T
# Get reflectivity from model builder object
ref = mb.get_refl()[slcy,:,:nz]
refo = dlut.resample(ref,[nxo,nzo],kind='linear').T

## Create a fake image
# Ricker wavelet parameters
nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
amp = 1.0; dly = 0.125
minf = 30.0; maxf = 60.0
# Compute ricker wavelet with random frequency 
f = rndut.randfloat(minf,maxf)
wav = ricker(nt,dt,f,amp,dly)
# Convolve with reflectivity
img = np.array([np.convolve(ref[ix,:],wav) for ix in range(nx)])[:,ns:1000+ns]
imgo = dlut.normalize(dlut.resample(img,[nxo,nzo],kind='linear')).T
# Create noise
nze = dlut.normalize(bandpass(np.random.rand(nzo,nxo)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
imgo += nze

## Plot reflectivity and image with labels
# Reflectivity
f4 = plt.figure(4,figsize=(14,7)); ax4 = f4.gca()
im4 = ax4.imshow(refo,cmap='gray',extent=[0,(nx-1)*10/1000.0,(nz+100-1)*5/1000.0,0.0])
ax4.set_xlabel('X (km)',fontsize=fsize)
ax4.set_ylabel('Z (km)',fontsize=fsize)
ax4.tick_params(labelsize=fsize)
plt.savefig('./fig/reffaults.png',bbox_inches='tight',dpi=150,transparent=True)
plt.close()

# Image
f5 = plt.figure(5,figsize=(14,7)); ax5 = f5.gca()
im5 = ax5.imshow(imgo,cmap='gray',extent=[0,(nx-1)*10/1000.0,(nz+100-1)*5/1000.0,0.0])
ax5.set_xlabel('X (km)',fontsize=fsize)
ax5.set_ylabel('Z (km)',fontsize=fsize)
ax5.tick_params(labelsize=fsize)
plt.savefig('./fig/imgfaults.png',bbox_inches='tight',dpi=150,transparent=True)
plt.close()

# Create mask
mask = np.ma.masked_where(lblo == 0, lblo)
cmap = colors.ListedColormap(['red','white'])
# Reflectivity with label
f6 = plt.figure(6,figsize=(14,7)); ax6 = f6.gca()
ax6.imshow(refo,cmap='gray',extent=[0,(nx-1)*10/1000.0,(nz+100-1)*5/1000.0,0.0])
ax6.imshow(mask,cmap,extent=[0,(nx-1)*10/1000.0,(nz+100-1)*5/1000.0,0.0])
ax6.set_xlabel('X (km)',fontsize=fsize)
ax6.set_ylabel('Z (km)',fontsize=fsize)
ax6.tick_params(labelsize=fsize)
plt.savefig('./fig/reffaultspos.png',bbox_inches='tight',dpi=150,transparent=True)
plt.close()

# Image with label
f7 = plt.figure(7,figsize=(14,7)); ax7 = f7.gca()
ax7.imshow(imgo,cmap='gray',extent=[0,(nx-1)*10/1000.0,(nz+100-1)*5/1000.0,0.0])
ax7.imshow(mask,cmap,extent=[0,(nx-1)*10/1000.0,(nz+100-1)*5/1000.0,0.0])
ax7.set_xlabel('X (km)',fontsize=fsize)
ax7.set_ylabel('Z (km)',fontsize=fsize)
ax7.tick_params(labelsize=fsize)
plt.savefig('./fig/imgfaultspos.png',bbox_inches='tight',dpi=150,transparent=True)
plt.close()

plt.show()

