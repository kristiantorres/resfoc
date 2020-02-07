import velocity.mdlbuild as mdlbuild
import scaas.noise_generator as noise_generator
from scaas.gradtaper import build_taper_ds
from utils.ptyprint import progressbar
import numpy as np
from deeplearn.utils import resample, thresh
import matplotlib.pyplot as plt

nx = 1000; ox=0.0; dx=25.0
ny = 200; oy=0.0; dy=25.0
dz = 12.5

mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)

### Set the random parameters for building the model
nlayer = 20
props = mb.vofz(nlayer,1600,5000)
print(np.min(props),np.max(props))

#plt.figure(1)
#plt.plot(props);
#plt.ylim([1000,5500])
#plt.show()

thicks = np.random.randint(40,61,nlayer)

#TODO: change the vertical variation parameters
#      layer, layer_rand, dev_layer so they change
#      from deposit to deposit

sqlyrs = sorted(mb.findsqlyrs(3,nlayer,5))
csq = 0

dlyr = 0.1
for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
  #print("Vel=%f Thick=%d"%(props[ilyr],thicks[ilyr]))
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],band2=0.01,band3=0.05,dev_pos=0.0,layer=150,layer_rand=0.00,dev_layer=dlyr)
  # Random folding
  if(ilyr in sqlyrs):
    if(sqlyrs[csq] < 15):
      amp = np.random.rand()*(3000-500) + 500
      mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
      #print("lyr=%d amp=%f"%(ilyr,amp))
    elif(sqlyrs[csq] >= 15 and sqlyrs[csq] < 18):
      amp = np.random.rand()*(1800-500) + 500
      mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
      #print("lyr=%d amp=%f"%(ilyr,amp))
    else:
      amp = np.random.rand()*(500-300) + 300
      mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
    csq += 1

# Water deposit
mb.deposit(1480,thick=50,layer=150,dev_layer=0.0)
# Trim model before faulting
mb.trim(0,1100)

# Build faults
# Probably at most 4 large faults and at least one
# Probably at least 5 small faults and at most 10
# Should put in even smaller faults near the surface

#mb.smallfault_block(nfault=5,azim=180.0,begz=0.25,begx=0.5,begy=0.5)
#mb.largefault_block(nfault=3,azim=0.0  ,begz=0.65,begx=0.3,begy=0.5)
#mb.smallgraben_block(azim=0.0,begz=0.25,begx=0.25,begy=0.5)
#mb.tinyfault_block(nfault=5,azim=0.0,begz=0.1,begx=0.3,begy=0.3)

#mb.largegraben_block(azim=0.0,begz=0.6,begx=0.3,begy=0.5)
#mb.sliding_block(nfault=3,azim=0.0,begz=0.3,begx=0.3,begy=0.5)
#mb.tinyfault_block(nfault=5,azim=180.0,begz=0.15,begx=0.7,begy=0.3)

#mb.smallhorstgraben_block(azim=0.0,begz=0.5)
mb.largehorstgraben_block(azim=0.0,begz=0.7)
mb.sliding_block(nfault=3,azim=0.0,begz=0.3,begx=0.3,begy=0.5)
mb.tinyfault_block(nfault=5,azim=0.0,begz=0.1,begx=0.3,begy=0.3)

# Get model
vel = mb.vel.T
nz = vel.shape[0]

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

