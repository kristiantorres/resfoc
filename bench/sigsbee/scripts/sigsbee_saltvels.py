import inpout.seppy as seppy
import numpy as np
from scaas.velocity import create_randomptbs_loc
import matplotlib.pyplot as plt

sep = seppy.sep()

vaxes,vel = sep.read_file("sigsbee_vel.H")

[nz,nx] = vaxes.n; [dz,dx] = vaxes.d; [oz,ox] = vaxes.o

vel = vel.reshape(vaxes.n,order='F')

# Select the two points for filling in the salt
# Upper point
#uz = 473; ux = 730
uz = 352; ux = 565
# Lower point
lz = 563; lx = 855

# Construct a line
dz = lz-uz; dx = lx-ux
m = dz/dx;
b = lz - m*lx

x = np.linspace(0,nx-1,nx)
z = np.linspace(0,nz-1,nz)

linez = x*m + b

velout = np.copy(vel)
velsalt = 4.51
for iz in range(nz):
  for ix in range(nx):
    # If point is in region, set to salt velocity
    myz = ix*m + b
    if(iz < myz and iz > uz and ix >= ux and ix <= lx):
      velout[iz,ix] = velsalt

vmin = 1.5; vmax = 4.51
plt.figure()
plt.imshow(vel,cmap='jet',vmin=vmin,vmax=vmax)
#plt.scatter(lx,lz)
#plt.scatter(ux,uz)
#plt.plot(x,linez)
plt.figure()
plt.imshow(velout,cmap='jet',vmin=vmin,vmax=vmax)
#plt.scatter(lx,lz)
#plt.scatter(ux,uz)
plt.show()

#sep.write_file("sigsbee_velsaltw2.H",velout ,ds=vaxes.d,os=vaxes.o)

