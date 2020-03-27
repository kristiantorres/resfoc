import velocity.mdlbuild as mdlbuild
import numpy as np
import inpout.seppy as seppy
import matplotlib.pyplot as plt

nx = 600; ox=0.0; dx=25.0
ny = 600; oy=0.0; dy=25.0
dz = 25

mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)
mb.deposit(velval=4000,thick=135,band2=0.01,band3=0.05,dev_pos=0.1,layer=25,layer_rand=0.3,dev_layer=0.3)

# Extract values from hypercube
vel = mb.vel.T
nz = vel.shape[0]

print("Actual nz=%d"%(vel.shape[0]))

# Write the velocity
sep = seppy.sep([])
vaxes = seppy.axes([nz,nx,ny],[0.0,ox,oy],[dz,dx,dy])
sep.write_file(None,vaxes,vel,ofname='me.H')

