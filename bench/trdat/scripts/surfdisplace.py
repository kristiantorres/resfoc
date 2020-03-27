import numpy as np
import matplotlib.pyplot as plt

def dist(x,x0,y,y0,lx=500,ly=500):
  return np.sqrt( ((x-x0)/lx)**2 + ((y-y0)/ly)**2 )

# Create a meshgrid
nx = 128; ox = 0.0; dx = 10.0
ny = 128; oy = 0.0; dy = 10.0

x0 = 64*dx; y0 = 64*dx

d = np.zeros([ny,nx])

dmax = 10

for iy in range(ny):
  y = oy + iy*dy
  for ix in range(nx):
    x = ox + ix*dx
    rxy = dist(x,x0,y,y0,lx=500,ly=1000)
    scale = ((1+rxy)**2)/4 - rxy**2
    if(scale >= 0):
      d[iy,ix] = 2*dmax*(1-rxy) * np.sqrt(scale)
    else:
      d[iy,ix] = 0.0


plt.imshow(d,cmap='jet')
plt.show()
