import velocity.mdlbuild as mdlbuild
from velocity.structure import vel_structure
from deeplearn.utils import resample
import numpy as np
import inpout.seppy as seppy
import matplotlib.pyplot as plt

nx = 1000; ox=0.0; dx=25.0
ny = 200; oy=0.0; dy=25.0
dz = 25

### Set the random parameters for building the model
## Define propagation velocities

nlayer = 20
props = np.zeros(nlayer)

props = np.linspace(5000,1500,nlayer)

thicks = np.zeros(nlayer,dtype='int32') + int(50)

#TODO: change the vertical variation parameters
#      layer, layer_rand, dev_layer so they change
#      from deposit to deposit

mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)
for ilyr in range(nlayer):
  print("Vel=%f Thick=%d"%(props[ilyr],thicks[ilyr]))
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],band2=0.01,band3=0.05,dev_pos=0.0,layer=150,layer_rand=0.00,dev_layer=0.10)

mb.largegraben_block(azim=0.0,begz=0.6,begx=0.3,begy=0.5,xdir=True)
mb.sliding_block(nfault=3,azim=0.0,begz=0.3,begx=0.3,begy=0.5,xdir=True)

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
lblresm = resample(lblwind,[1024,512],kind='linear')
plt.imshow(lblresm.T,cmap='jet')
plt.show()

