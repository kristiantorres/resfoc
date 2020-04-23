import numpy as np
import matplotlib.pyplot as plt

nx = 1000; nz = 1000
img = np.zeros([nx,nz])

dx = 25; dz = 12.5

r = 7071.067811865475

xcenter = 7500
zcenter = -1250

for ix in range(nx):
  x = ix*dx - xcenter
  for iz in range(nz):
    z = iz*dz - zcenter
    pr = np.sqrt(x*x + z*z)
    if(np.abs(pr - r) < 20):
      img[ix,iz] = 1.0

plt.imshow(img.T,extent=[0,nx*dx,nz*dz,0])
plt.scatter(xcenter,zcenter)
#plt.scatter(xbeg,zbeg)
plt.show()

