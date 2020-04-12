import velocity.mdlbuild as mdlbuild
from velocity.structure import vel_structure
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

props = np.linspace(5000,1500,nlayer)

thicks = np.zeros(nlayer,dtype='int32') + int(50)

#TODO: change the vertical variation parameters
#      layer, layer_rand, dev_layer so they change
#      from deposit to deposit

dlyr = 0.1
mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)
for ilyr in range(nlayer):
  print("Vel=%f Thick=%d"%(props[ilyr],thicks[ilyr]))
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],band2=0.01,band3=0.05,dev_pos=0.0,layer=150,layer_rand=0.00,dev_layer=dlyr)
  if(ilyr == 5 or ilyr == 14):
    #mb.squish(amp=1500,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
    mb.squish(amp=3000,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
  if(ilyr == 18):
    mb.squish(amp=900,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')

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

