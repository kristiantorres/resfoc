import numpy as np
import velocity.mdlbuild as mdlbuild
from scaas.noise_generator import perlin
import matplotlib.pyplot as plt
from utils.ptyprint import progressbar
from scipy.ndimage import map_coordinates
import inpout.seppy as seppy
from scaas.trismooth import smooth

sep = seppy.sep()

nx = 1000; ny = 10; slcy = int(ny/2)
minvel = 1600; maxvel = 3000; nlayer = 6
dx = dy = 25; dz = 12.5

# Model building object
mb = mdlbuild.mdlbuild(nx,dx,ny=ny,dy=dx,dz=dz,basevel=3000)
nzi = 1000 # internal size is 1000

# Propagation velocities
props = np.linspace(maxvel,minvel,21)

# Specify the thicknesses
thicks = np.random.randint(40,61,nlayer)
#thicks = [47, 49, 51, 47, 46, 49]
#thicks = [56, 44, 46, 51, 44, 60]

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
shf1d = perlin(x=np.linspace(0,3,nn), octaves=2, period=80, persist=0.6, ncpu=1)
shf1d -= np.mean(shf1d); shf1d *= 10*amp
sep.write_file("shf1d.H",shf1d)
#_,shf1d = sep.read_file("shf1d.H")
shf = np.ascontiguousarray(np.tile(shf1d,(nn,1)).T).astype('float32')
# Find the maximum shift to be applied
pamp = np.max(np.abs(shf1d))
maxshift = int(pamp/dz) + 5
# Expand the model
nzot = nzin + 2*maxshift
velot = np.zeros([ny,nx,nzot],dtype='float32')
lyrot = np.zeros([ny,nx,nzot],dtype='int32')
mb.ec8.expand(maxshift,maxshift,nzin,lyr,vel,nzot,lyrot,velot)

# Arrays must be same size
velot2 = np.copy(velot)
lyrot2 = np.copy(lyrot)

# Shifts to be computed
coords = np.zeros([3,*velot2.shape],dtype='float32')

# Compute shifts
mb.ec8.squish_shifts(nzot,shf,1,90.0,0.4,0.0,0.0,coords[0],coords[1],coords[2])
#mb.ec8.squish(nzin,lyr,vel,shf,1,90.0,pamp,0.4,0.0,0.0,nzot,lyrot,velot)

# Perform mapping
velot3 = map_coordinates(velot2,coords,order=3,mode='constant',cval=-1)
lyrot3 = map_coordinates(lyrot2,coords,order=0,mode='constant',cval=-1)


velot4 = np.copy(velot3)
lyrot4 = np.copy(lyrot3)
#lyrot4 = np.zeros(lyrot3.shape,dtype='int32')

idx = velot3 <= 0
velot3[idx] = -1

mb.ec8.fill_top_bottom(nzot,pamp,3000.0,lyrot4,velot4)

idx = velot4 <= 0
velot4[idx] = -1

itype = 'none'
vmin = np.min(vel[slcy]); vmax = np.max(vel[slcy])
#plt.figure(1)
#plt.imshow(velot[slcy].T,cmap='jet',interpolation=itype,vmin=vmin,vmax=vmax)
plt.figure(2)
plt.imshow(lyrot3[slcy].T,cmap='jet',interpolation=itype)
plt.figure(3)
plt.imshow(velot3[slcy].T,cmap='jet',interpolation=itype,vmin=vmin,vmax=vmax)
plt.figure(4)
plt.imshow(lyrot4[slcy].T,cmap='jet',interpolation=itype)
plt.figure(5)
plt.imshow(velot4[slcy].T,cmap='jet',interpolation=itype)
plt.show()

print(velot.shape,velot4.shape)

#mb.vel = velot.astype('float32')
#mb.lyr = lyrot.astype('int32')
mb.vel = velot4.astype('float32')
mb.lyr = lyrot3.astype('int32')

# Add another deposit
mb.deposit(velval=2200,thick=50,band2=0.01,band3=0.05,dev_pos=0.0,layer=50,layer_rand=0.00,dev_layer=dlyr)

plt.figure(6)
plt.imshow(mb.vel[slcy].T,cmap='jet',interpolation=itype)
#plt.figure(7)
#plt.imshow(smooth(mb.vel,rect1=1,rect2=20,rect3=1)[slcy].T,cmap='jet',interpolation=itype)
plt.show()


