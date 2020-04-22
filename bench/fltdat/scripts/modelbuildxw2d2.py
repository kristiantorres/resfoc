import numpy as np
from scipy import interpolate
from scaas import noise_generator
import matplotlib.pyplot as plt

def dist(x,x0,lx=500):
  return np.sqrt( ((x-x0)/lx)**2 )

def rotate(x,z,x0,z0,thetar):
  """ Converts local coordinates to global coordinates """
  xr = (np.cos(thetar)*x + np.sin(thetar)*z) + x0
  zr = (np.sin(thetar)*x - np.cos(thetar)*z) + z0

  return xr,zr

# Create a meshgrid
nx = 512; ox = 0.0; dx = 10.0
nz = 512; oz = 0.0; dz = 10.0

x = np.linspace(ox,ox+(nx-1)*dx,nx)
z = np.linspace(oz,oz+(nz-1)*dz,nz)

posz = 256; posx = 256

# Put in a vertical fault
img = np.zeros([nx,nz])
img[posx,:] = 1

z0 = posz*dz; x0 = posx*dx
dmax = 100

crdx = np.zeros(nz) + posx*dx
crdz = np.linspace(oz,oz+(nz-1)*dz,nz)

# Create the vertical fault surface
for iz in range(nz):
  z = oz + iz*dz - z0
  rz = dist(z,0.0,lx=2000)
  scale = ((1+rz)**2)/4 - rz**2
  if(scale >= 0):
    img[posx,iz] = 2*dmax*(1-rz) * np.sqrt(scale)
  else:
    img[posx,iz] = 0.0

# Perturb the fault surface
ptb = noise_generator.perlin(x=np.linspace(0,2,nz), octaves=3, period=80, Ngrad=80, persist=0.3, ncpu=1)
ptb -= np.mean(ptb);

crdxp = x0 + 1000*ptb
ixp = (crdxp/dx + 0.5).astype('int')

#TODO: should apply the shifts with linear interpolation here
nimg = np.zeros(img.shape)
for iz in range(nz):
  nimg[ixp[iz],iz] = img[posx,iz]

# Compute the displacements
lam = 0.5; gam = 200*dx
disp = np.zeros([nx,nz])
for ix in range(nx):
  x = ox + ix*dx - x0
  for iz in range(nz):
    z = oz + iz*dz - z0
    if(x >= (crdxp[iz]-x0) and x <= gam):
      disp[ix,iz] = lam * nimg[ixp[iz],iz] * (1-np.abs(x)/gam)**2
    elif(x <= (crdxp[iz]-x0) and x >= -gam):
      disp[ix,iz] = (lam-1) * nimg[ixp[iz],iz] * (1-np.abs(x)/gam)**2

# Compute the z displacement

## Rotate into the other coordinate system
#dispr = np.zeros(disp.shape)
#for ix in range(nx):
#  x = ox + ix*dx - x0
#  for iz in range(nz):
#    z = oz + iz*dz - z0
#    # Apply rotation
#    xr,zr = rotate(x,z,x0,z0,theta*(np.pi/180))
#    # Compute indices
#    ixr = int(xr/dx+0.5); izr = int(zr/dz+0.5);
#    print(ixr,izr)
#    if(ixr >= 0 and ixr < nx and izr >=0 and izr < nz):
#      dispr[ixr,izr] = disp[ix,iz]

theta = -30
dxr,dzr = rotate(dx,dz,0.0,0.0,theta*(np.pi/180))

# Rotate into the other coordinate system
dispr = np.zeros(disp.shape)
for ixr in range(nx):
  xr = ox + ixr*dx - x0
  for izr in range(nz):
    zr = oz + izr*dz - z0
    # Apply rotation
    x,z = rotate(xr,zr,x0,z0,theta*(np.pi/180))
    # Compute indices
    ix = int(x/dx+0.5); iz = int(z/dz+0.5);
    if(ix >= 0 and ix < nx and iz >=0 and iz < nz):
      dispr[ixr,izr] = disp[ix,iz]
      #TODO: rotate the surface as well

plt.figure(1)
plt.imshow(img.T,cmap='jet')

plt.figure(2)
plt.imshow(nimg.T,cmap='jet')

plt.figure(3)
plt.imshow(disp.T,cmap='jet')

plt.figure(4)
plt.imshow(dispr.T,cmap='jet')
plt.show()

