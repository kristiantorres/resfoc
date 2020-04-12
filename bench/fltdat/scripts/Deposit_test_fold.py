import velocity.mdlbuild as mdlbuild
import numpy as np
import matplotlib.pyplot as plt

nx = 600; ox=0.0; dx=25.0
ny = 600; oy=0.0; dy=25.0
dz = 25

mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)

mb.deposit(velval=4000,thick=100,layer=50,dev_layer=.2,layer_rand=0.0)
mb.deposit(velval=3000,thick=100,layer=50,dev_layer=.2,layer_rand=0.0)
mb.squish(amp=500,azim=90.0,lam=0.7,rinline=0.0,rxline=0.0,mode='perlin')
mb.squish(amp=200,azim=90.0,lam=0.7,rinline=0.0,rxline=0.0,mode='perlin')
#mb.squish(amp=200,azim=90.0,lam=0.2,rinline=0.0,rxline=0.0)
#mb.squish(max=200,random_inline=0.0,random_crossline=0.0,azimuth=90.0,wavelength=0.2)

# Extract values from hypercube
vel = mb.vel.T
nz = vel.shape[0]

plt.figure(1)
plt.imshow(vel[:,:,100],cmap='jet')
plt.show()

