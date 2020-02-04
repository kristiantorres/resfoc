import velocity.mdlbuild as mdlbuild
import scaas.noise_generator as noise_generator
from scaas.gradtaper import build_taper_ds
import numpy as np
import inpout.seppy as seppy
from deeplearn.utils import resample, thresh
import matplotlib.pyplot as plt

nx = 1000; ox=0.0; dx=25.0
ny = 200; oy=0.0; dy=25.0
dz = 12.5

### Set the random parameters for building the model
## Define propagation velocities

nlayer = 20
props = np.zeros(nlayer)

props = np.linspace(5000,1600,nlayer)

# Generate smooth function to perturb the props
npts = 2; octaves = 3; persist = 0.3
ptb = noise_generator.perlin(x=np.linspace(0,npts,nlayer), octaves=octaves, period=80, Ngrad=80, persist=persist, ncpu=2)
ptb -= np.mean(ptb);
tap,_ = build_taper_ds(1,20,1,5,15,19)
props += 5000*(ptb*tap)

print(np.min(props),np.max(props))

plt.figure(1)
plt.plot(ptb)
plt.figure(2)
plt.plot(props);
plt.ylim([1000,5500])
plt.show()

thicks = np.zeros(nlayer,dtype='int32') + int(50)
thicks = np.random.randint(40,61,nlayer)

#TODO: change the vertical variation parameters
#      layer, layer_rand, dev_layer so they change
#      from deposit to deposit

nsq = np.random.randint(1,4)
sqlyrs = np.random.randint(0,nlayer,nsq)

dlyr = 0.1
mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)
for ilyr in range(nlayer):
  print("Vel=%f Thick=%d"%(props[ilyr],thicks[ilyr]))
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],band2=0.01,band3=0.05,dev_pos=0.0,layer=150,layer_rand=0.00,dev_layer=dlyr)
  if(ilyr == 5 or ilyr == 14):
    amp = np.random.rand()*(3000-100) + 100
    mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
    #mb.squish(amp=3000,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
  if(ilyr == 18):
    amp = np.random.rand()*(1200-100) + 100
    mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
    #mb.squish(amp=1200,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')

mb.deposit(1480,thick=50,layer=150,dev_layer=0.0)
mb.trim(0,1000)

mb.smallfault_block(nfault=5,azim=180.0,begz=0.3,begx=0.5,begy=0.5,xdir=True)
mb.largefault_block(nfault=3,azim=0.0  ,begz=0.7,begx=0.3,begy=0.5,xdir=True)
mb.smallgraben_block(azim=0.0,begz=0.3,begx=0.25,begy=0.5,xdir=True)

# Extract values from hypercube
vel = mb.vel.T
nz = vel.shape[0]

print("Actual nz=%d"%(vel.shape[0]))

lbl = mb.get_label().T

plt.figure(1)
plt.imshow(vel[:,:,100],cmap='jet')

plt.figure(2)
plt.imshow(lbl[:,:,100],cmap='jet')
plt.show()

plt.figure(1)
velwind = mb.vel[100,:,:]
velresm = resample(velwind,[1024,512],kind='linear')
plt.imshow(velresm.T,cmap='jet')

plt.figure(2)
lblwind = mb.get_label()[100,:,:]
lblresm = thresh(resample(lblwind,[1024,512],kind='linear'),0)
plt.imshow(lblresm.T,cmap='jet')
plt.show()

