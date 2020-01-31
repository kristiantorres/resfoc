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
props = np.zeros(4)
# Bottom layer
prop1min = 4500
prop1max = 5500
props[0] = np.random.rand()*(prop1max - prop1min) + prop1min

# Second layer
prop2min = 3500
prop2max = 4500
props[1] = np.random.rand()*(prop2max - prop2min) + prop2min

# Third layer
prop3min = 2500
prop3max = 3500
props[2] = np.random.rand()*(prop3max - prop3min) + prop3min

# Fourth layer
prop4min = 1500
prop4max = 2500
props[3] = np.random.rand()*(prop4max - prop4min) + prop4min

mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)
stctr = vel_structure(nx)
for ilyr in range(4):
  tag = str(ilyr+1)
  print("Depositing: %d samples thick"%(stctr['thick'+tag]))
  mb.deposit(velval=props[ilyr],thick=stctr['thick'+tag],band2=0.01,band3=0.05,dev_pos=0.1,layer=25,layer_rand=0.3,dev_layer=0.3)

print("Faulting")
mb.fault(begx=0.5,begy=0.5,begz=0.5,daz=8000.0,dz=7000.0,azim=180.0,theta_die=12.0,theta_shift=4.0,dist_die=0.3,perp_die=0.5)
#mb.fault(begx=0.2,begy=0.5,begz=0.5,daz=8000.0,dz=7000.0,azim=180.0,theta_die=12.0,theta_shift=4.0,dist_die=0.3,perp_die=0.5)

# Water layer
mb.deposit(velval=1500,thick=10,dev_layer=0,layer_rand=0,layer=100,dev_pos=0.0)

# Extract values from hypercube
vel = mb.vel.T
nz = vel.shape[0]

lbl = mb.get_label().T
print(lbl.shape)

print("Actual nz=%d"%(vel.shape[0]))

# Write the velocity
sep = seppy.sep([])
vaxes = seppy.axes([nz,nx,ny],[0.0,ox,oy],[dz,dx,dy])
sep.write_file(None,vaxes,vel,ofname='me.H')
sep.write_file(None,vaxes,lbl,ofname='lbl.H')

