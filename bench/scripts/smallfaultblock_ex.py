import velocity.mdlbuild as mdlbuild
from velocity.structure import vel_structure
import numpy as np
import inpout.seppy as seppy
import matplotlib.pyplot as plt

nx = 600; ox=0.0; dx=25.0
ny = 600; oy=0.0; dy=25.0
dz = 25

### Set the random parameters for building the model
## Define propagation velocities
props = np.zeros(2)
thicks = np.zeros(2,dtype='int32')
devs = np.zeros(2,dtype='float32')
layers = np.zeros(2,dtype='float32')
# Bottom layer
props[0] = 3500
thicks[0] = 140
devs[0] = 0.2
layers[0] = 400

# Fourth layer
props[1] = 2500
thicks[1] = 150
devs[1] = 0.2
layers[1] = 150

mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)
for ilyr in range(2):
  tag = str(ilyr+1)
  print("Depositing: %d samples thick"%(thicks[ilyr]))
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],band2=0.01,band3=0.05,dev_pos=0.0,layer=layers[ilyr],layer_rand=0.0,dev_layer=devs[ilyr])

#TODO: test on a bigger model
mb.smallfault_block(nfault=5,azim=0.0,begz=0.3,begx=0.7,begy=0.5,xdir=True)

# Extract values from hypercube
vel = mb.vel.T
nz = vel.shape[0]

lbl = mb.get_label().T
print(lbl.shape)

print("Actual nz=%d"%(vel.shape[0]))

f,axarr = plt.subplots(1,2,figsize=(10,5))
axarr[0].imshow(vel[:-40,:,300],cmap='jet',vmin=1600,vmax=4100)
axarr[1].imshow(lbl[:-40,:,300],cmap='jet')
plt.show()

