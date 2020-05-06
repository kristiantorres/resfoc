import numpy as np
import velocity.mdlbuild as mdlbuild
from scaas.noise_generator import perlin
import matplotlib.pyplot as plt
from utils.ptyprint import progressbar
import inpout.seppy as seppy

sep = seppy.sep()

nx = 1000; ny = 200; slcy = int(ny/2)
minvel = 1600; maxvel = 3000; nlayer = 6
dx = dy = 25; dz = 12.5

# Model building object
mb = mdlbuild.mdlbuild(nx,dx,ny=200,dy=dx,dz=dz,basevel=3000)
nzi = 1000 # internal size is 1000

# Propagation velocities
props = np.linspace(maxvel,minvel,21)

# Specify the thicknesses
#thicks = np.random.randint(40,61,nlayer)
thicks = [47, 49, 51, 47, 46, 49]

dlyr = 0.05
for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],band2=0.01,band3=0.05,dev_pos=0.0,layer=50,layer_rand=0.00,dev_layer=dlyr)

# Water deposit
#mb.deposit(1480,thick=80,layer=150,dev_layer=0.0)

vel = mb.vel
lyr = mb.lyr

nzin = vel.shape[2]

# Compute shifts for squish
nn = 3*max(nx,ny)
shf = np.zeros([nn,nn],dtype='float32')

amp = 400
#shf1d = perlin(x=np.linspace(0,3,nn), octaves=3, period=80, persist=0.6, ncpu=1)
#shf1d -= np.mean(shf1d); shf1d *= 10*amp
#sep.write_file("shf1d.H",shf1d)
_,shf1d = sep.read_file("shf1d.H")
shf = np.ascontiguousarray(np.tile(shf1d,(nn,1)).T).astype('float32')
# Find the maximum shift to be applied
pamp = np.max(np.abs(shf1d))
maxshift = int(pamp/dz)
# Expand the model
nzot = nzin + 2*maxshift
velot = np.zeros([ny,nx,nzot],dtype='float32')
lyrot = np.zeros([ny,nx,nzot],dtype='int32')
mb.ec8.expand(maxshift,maxshift,nzin,lyr,vel,nzot,lyrot,velot)

print(pamp)

print(vel.shape)
velot2 = np.copy(velot)
lyrot2 = np.copy(lyrot)

print(velot.shape)
mb.ec8.squish_shifts(nzot,lyrot2,velot2,shf,1,90.0,pamp,0.4,0.0,0.0,lyrot,velot)
#mb.ec8.squish(nzin,lyr,vel,shf,1,90.0,pamp,0.4,0.0,0.0,nzot,lyrot,velot)

print(velot.shape)
#idx = velot == -1
#velot[idx] = 2500

plt.figure(1)
plt.imshow(vel[slcy].T,cmap='jet',interpolation='bilinear')
plt.figure(2)
plt.imshow(velot[slcy].T,cmap='jet',interpolation='bilinear')
plt.figure(3)
plt.imshow(lyrot[slcy].T,cmap='jet')
plt.show()

