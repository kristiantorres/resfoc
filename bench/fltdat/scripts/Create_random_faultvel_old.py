import velocity.mdlbuild as mdlbuild
import scaas.noise_generator as noise_generator
from scaas.gradtaper import build_taper_ds
from scaas.wavelet import ricker
from utils.ptyprint import progressbar
import utils.rand as rndut
import numpy as np
import deeplearn.utils as dlut
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

nx = 1000; ox=0.0; dx=25.0
ny = 200; oy=0.0; dy=25.0
dz = 12.5

mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)

### Set the random parameters for building the model
nlayer = 20
props = mb.vofz(nlayer,1600,5000)
print(np.min(props),np.max(props))

thicks = np.random.randint(40,61,nlayer)

sqlyrs = sorted(mb.findsqlyrs(3,nlayer,5))
csq = 0

dlyr = 0.1
for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],band2=0.01,band3=0.05,dev_pos=0.0,layer=150,layer_rand=0.00,dev_layer=dlyr)
  # Random folding
  if(ilyr in sqlyrs):
    if(sqlyrs[csq] < 15):
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
mb.trim(0,1100)

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

# Get model
vel = mb.vel.T
nz = vel.shape[0]

lbl = mb.get_label().T

plt.figure(1)
velwind = mb.vel[100,:,:1000]
velwind = gaussian_filter(velwind,sigma=0.8)
velresm = dlut.resample(velwind,[1024,512],kind='linear')
plt.imshow(velresm.T,cmap='jet')

plt.figure(2)
lblwind = mb.get_label()[100,:,:1000]
lblresm = dlut.thresh(dlut.resample(lblwind,[1024,512],kind='linear'),0)
plt.imshow(lblresm.T,cmap='jet')

#velwindsm = gaussian_filter(velwind,sigma=20)
#velwindsmresm = dlut.resample(velwindsm,[1024,512],kind='linear')
#plt.figure(3)
#plt.imshow(velwindsmresm.T,cmap='jet')

# Reflectivity and fault label
#dvel = velwind - velwindsm
#dvelresm = dlut.resample(dvel,[1024,512],kind='linear')
#plt.figure(4)
#plt.imshow(dlut.normalize(dvelresm.T),cmap='gray')

#dlut.plotseglabel(dvelresm.T,lblresm.T)

#plt.figure(6)
refl = mb.get_refl()[100,:,:1000]
#import inpout.seppy as seppy
#sep = seppy.sep([])
#axes = seppy.axes([1000,1000],[0.0,0.0],[1.0,1.0])
#sep.write_file(None,axes,refl.T,ofname='refl.H')
#reflresm = dlut.resample(refl,[1024,512],kind='linear')
#plt.imshow(dlut.normalize(reflresm.T),cmap='gray')

#dlut.plotseglabel(reflresm.T,lblresm.T)

# Convolve with wavelet
nt = 250; dt = 0.001; f = 50.0; amp = 1.0; dly=0.125
ns = int(nt/2)
wav = ricker(nt,dt,f,amp,dly)
img = np.array([np.convolve(refl[ix,:],wav) for ix in range(nx)])[:,ns:1000+ns]
imgresm = dlut.resample(img,[1024,512],kind='linear')
plt.figure(8)
plt.imshow(dlut.normalize(imgresm.T),cmap='gray')
dlut.plotseglabel(imgresm.T,lblresm.T)

plt.show()

