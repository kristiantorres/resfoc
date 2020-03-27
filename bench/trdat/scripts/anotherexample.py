import velocity.mdlbuild as mdlbuild
from velocity.structure import vel_structure
from utils.ptyprint import progressbar,create_inttag
import numpy as np
import matplotlib.pyplot as plt

nx = 600; ox=0.0; dx=25.0
ny = 600; oy=0.0; dy=25.0
dz = 25

### Set the random parameters for building the model
## Define propagation velocities
props = np.zeros(2)
thicks = np.zeros(2,dtype='int32')
devs = np.zeros(2,dtype='float32')
lrands = np.zeros(2,dtype='float32')
layers = np.zeros(2,dtype='float32')
# Bottom layer
props[0] = 3500
thicks[0] = 140
# Control the variation within a layer
devs[0] = 0.2
layers[0] = 150

# Fourth layer
props[1] = 2500
thicks[1] = 150
devs[1] = 0.25
layers[1] = 50

# Build the layered model
mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)
for ilyr in range(2):
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],band2=0.01,band3=0.05,dev_pos=0.0,layer=layers[ilyr],layer_rand=0.05,dev_layer=devs[ilyr])

# Make a copy of the layerd model
velcopy = mb.vel.copy()

dz = 2000
# Build all of the dazs to be used
mindaz = 500; maxdaz = 11000; ddaz = 500
dazs = np.arange(mindaz,maxdaz,ddaz)
ndaz = len(dazs)
# Loop over all parameters
for idaz in progressbar(dazs, "Example: ", 40):
  # Reset the model and the label
  mb.vel = velcopy.copy(); mb.lbl[:] = 0
  # Put in fault
  mb.fault(begx=0.5,begy=0.5,begz=0.5,daz=7000,dz=2000,azim=180.0,theta_die=12.0,theta_shift=4.0,dist_die=0.3,perp_die=0.5)
  mb.fault(begx=0.4,begy=0.5,begz=0.5,daz=7000,dz=2000,azim=180.0,theta_die=12.0,theta_shift=4.0,dist_die=0.3,perp_die=0.5)
  mb.fault(begx=0.55,begy=0.5,begz=0.55,daz=6000,dz=1000,azim=0.0,theta_die=12.0,theta_shift=4.0,dist_die=0.3,perp_die=0.5)

  # Extract values from hypercube
  vel = mb.vel.T
  nz = vel.shape[0]

  # Get label
  lbl = mb.get_label().T

  # Plot model and label
  f,axarr = plt.subplots(1,2,figsize=(15,7))
  axarr[0].imshow(vel[:-40,:,300],cmap='jet',vmin=1600,vmax=4100)
  axarr[0].set_title('daz=%f dz=%f'%(idaz,dz))
  axarr[1].imshow(lbl[:-40,:,300],cmap='jet')
  axarr[1].set_title('daz=%f dz=%f'%(idaz,dz))
  #plt.savefig(('./fig/dazmovie/daz-%s.png'%(create_inttag(idaz,maxdaz))),bbox_inches='tight')
  plt.show()

