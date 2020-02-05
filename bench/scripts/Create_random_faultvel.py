import velocity.mdlbuild as mdlbuild
import scaas.noise_generator as noise_generator
from scaas.gradtaper import build_taper_ds
import numpy as np
import inpout.seppy as seppy
from deeplearn.utils import resample, thresh
import matplotlib.pyplot as plt

def findsqlyrs(nlyrs,ntot,mindist):
  """
  Finds layer indices to squish. Makes sure that they layers
  are dist indices apart and are not the same

  Parameters:
    nlyrs - number of layers to squish
    ntot  - total number of layers to be deposited
    mindist - minimum distance between lay

  """
  # Get the first layer
  sqlyrs = []
  sqlyrs.append(np.random.randint(0,ntot))
  # Loop until all layers are found
  while(len(sqlyrs) < nlyrs):
    lidx = np.random.randint(0,ntot)
    # Compute distances
    nsq = len(sqlyrs)
    sqdist = np.zeros(nsq,dtype='int32')
    for isq in range(nsq):
      sqdist[isq] = np.abs(lidx - sqlyrs[isq])
    if(np.all(sqdist >= mindist)):
      sqlyrs.append(lidx)

  return sqlyrs

nx = 1000; ox=0.0; dx=25.0
ny = 200; oy=0.0; dy=25.0
dz = 12.5

### Set the random parameters for building the model
## Define propagation velocities

# Generate smooth function to perturb the props
nlayer = 20
props = np.zeros(nlayer)
props = np.linspace(5000,1600,nlayer)
npts = 2; octaves = 3; persist = 0.3
ptb = noise_generator.perlin(x=np.linspace(0,npts,nlayer), octaves=octaves, period=80, Ngrad=80, persist=persist, ncpu=2)
ptb -= np.mean(ptb);
tap,_ = build_taper_ds(1,20,1,5,15,19)
props += 5000*(ptb*tap)

print(np.min(props),np.max(props))

#plt.figure(1)
#plt.plot(ptb)
#plt.figure(2)
#plt.plot(props);
#plt.ylim([1000,5500])
#plt.show()

thicks = np.zeros(nlayer,dtype='int32') + int(50)
thicks = np.random.randint(40,61,nlayer)

#TODO: change the vertical variation parameters
#      layer, layer_rand, dev_layer so they change
#      from deposit to deposit

sqlyrs = sorted(findsqlyrs(3,nlayer,5))
print(sqlyrs)
csq = 0

dlyr = 0.1
mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)
for ilyr in range(nlayer):
  print("Vel=%f Thick=%d"%(props[ilyr],thicks[ilyr]))
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],band2=0.01,band3=0.05,dev_pos=0.0,layer=150,layer_rand=0.00,dev_layer=dlyr)
  # Random folding
  if(ilyr in sqlyrs):
    if(sqlyrs[csq] < 15):
      amp = np.random.rand()*(3000-500) + 500
      mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
      print("lyr=%d amp=%f"%(ilyr,amp))
    elif(sqlyrs[csq] >= 15 and sqlyrs[csq] < 18):
      amp = np.random.rand()*(1800-500) + 500
      mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
      print("lyr=%d amp=%f"%(ilyr,amp))
    else:
      amp = np.random.rand()*(500-300) + 300
      mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
    csq += 1

# Water deposit
mb.deposit(1480,thick=50,layer=150,dev_layer=0.0)
# Trim model before faulting
mb.trim(0,1100)

mb.smallfault_block(nfault=5,azim=180.0,begz=0.25,begx=0.5,begy=0.5,xdir=True)
mb.largefault_block(nfault=3,azim=0.0  ,begz=0.65,begx=0.3,begy=0.5,xdir=True)
mb.smallgraben_block(azim=0.0,begz=0.25,begx=0.25,begy=0.5,xdir=True)

# Get model
vel = mb.vel.T
nz = vel.shape[0]

#print("Actual nz=%d"%(vel.shape[0]))

lbl = mb.get_label().T

plt.figure(1)
plt.imshow(vel[:1000,:,100],cmap='jet')
plt.figure(2)
plt.imshow(lbl[:1000,:,100],cmap='jet')
plt.show()

plt.figure(1)
velwind = mb.vel[100,:,:1000]
velresm = resample(velwind,[1024,512],kind='linear')
plt.imshow(velresm.T,cmap='jet')

plt.figure(2)
lblwind = mb.get_label()[100,:,:1000]
lblresm = thresh(resample(lblwind,[1024,512],kind='linear'),0)
plt.imshow(lblresm.T,cmap='jet')
plt.show()

