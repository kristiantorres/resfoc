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

#TODO: make movies of dz and daz
#TODO: Also, make movies of the other parameters
print("Faulting")
mb.fault(begx=0.8,begy=0.5,begz=0.3,daz=8000.0,dz=7000.0,azim=180.0,theta_die=12.0,theta_shift=4.0,dist_die=0.3,perp_die=0.5)
mb.fault(begx=0.25,begy=0.5,begz=0.3,daz=6000.0,dz=3000.0,azim=180.0,theta_die=12.0,theta_shift=4.0,dist_die=0.3,perp_die=0.5)
mb.fault(begx=0.35,begy=0.5,begz=0.3,daz=6000.0,dz=3000.0,azim=0.0,theta_die=12.0,theta_shift=4.0,dist_die=0.3,perp_die=0.5)

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

## Write the velocity
#sep = seppy.sep([])
#vaxes = seppy.axes([nz,nx,ny],[0.0,ox,oy],[dz,dx,dy])
#sep.write_file(None,vaxes,vel,ofname='me.H')
#sep.write_file(None,vaxes,lbl,ofname='lbl.H')

