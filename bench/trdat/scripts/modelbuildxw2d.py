import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scaas import noise_generator

def dist(x,x0,lx=500):
  return np.sqrt( ((x-x0)/lx)**2 )

def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx,array[idx]

def glob2loc(x,z,x0,z0,thetar):
  """ Converts global coordinates to local coordinates """
  xl = np.cos(thetar)*(x-x0) + np.sin(thetar)*(z-z0)
  zl = np.sin(thetar)*(x-x0) - np.cos(thetar)*(z-z0)

  return xl,zl

def loc2glob(x,z,x0,z0,thetar):
  """ Converts local coordinates to global coordinates """
  xb = (np.cos(thetar)*x + np.sin(thetar)*z) + x0
  zb = (np.sin(thetar)*x - np.cos(thetar)*z) + z0

  return xb,zb

# Create a meshgrid
nx = 512; ox = 0.0; dx = 10.0
nz = 512; oz = 0.0; dz = 10.0

x = np.linspace(ox,ox+(nx-1)*dx,nx)
z = np.linspace(oz,oz+(nz-1)*dz,nz)

# Create an image
img = np.zeros([nx,nz],dtype='float32')

# First create the center coordinate
cx = 256; cz = 256
x0 = cx*dx; z0 = cz*dz

# Choose the dip angle
theta = 45

# Convert to radians
thetar = np.pi/180*theta

dmax = 100
ptsx = []
ptsz = []

# Loop over each point
for ix in range(nx):
  x = ox + ix*dx
  for iz in range(nz):
    z = oz + iz*dz
    # Rotate the point into a local coordinate system
    xl,zl = glob2loc(x,z,x0,z0,thetar)
    if(np.abs(zl) < 5):
      ptsx.append(xl); ptsz.append(zl)
      rx = dist(xl,0.0,lx=2000)
      scale = ((1+rx)**2)/4 - rx**2
      img[ix,iz] = 1.0
      if(scale >= 0):
        img[ix,iz] = 2*dmax*(1-rx) * np.sqrt(scale)
      else:
        img[ix,iz] = 0.0

# Convert to numpy arrays
ptsx = np.array(ptsx); ptsz = np.array(ptsz)

plt.figure(1)
plt.imshow(img,cmap='jet')

# Transform back to global coordinates
xb,zb = loc2glob(ptsx,ptsz,x0,z0,thetar)

#TODO: add the spline capability
# Smoothly Perturb the surface
ptb = noise_generator.perlin(x=np.linspace(0,2,len(zb)), octaves=3, period=80, Ngrad=80, persist=0.3, ncpu=1)
ptb -= np.mean(ptb);

mag = 0
ptszp = ptsz + mag*ptb

#TODO: linearly interpolate instead here
xbi = (xb/dx).astype('int')
zbi = (zb/dz).astype('int')
zbip = ((zb+mag*ptb)/dz).astype('int')

#XXX: need to convert to local coordinates before indexing
nimg = np.zeros(img.shape)
for i in range(len(xbi)):
  if(xbi[i] < nx and xbi[i] >= 0 and zbip[i] < nz and zbip[i] >= 0):
    #print(xbi[i],zbip[i])
    nimg[xbi[i],zbip[i]] = img[xbi[i],zbi[i]]

plt.figure(5)
plt.imshow(nimg,cmap='jet')

# Work only in local coordinates
iptsx  = (ptsx/dx).astype('int')
iptsz  = (ptsz/dz).astype('int')
iptszp = ((ptszp)/dz).astype('int')

print('start')
# Loop again over points in x and z
bini = np.zeros(img.shape)
disp = np.zeros(img.shape)
for ix in range(nx):
  x = ox + ix*dx
  for iz in range(nz):
    z = oz + iz*dz
    xl,zl = glob2loc(x,z,x0,z0,thetar)
    # Find the x
    idx,xinl = find_nearest(ptsx,xl)
    zinl = ptszp[idx]
    # Convert surface points to global
    xinb,zinb = loc2glob(xinl,zinl,x0,z0,thetar)
    #print(int(xinb/dx),int(zinb/dz+0.5))
    if( zl > (-(zinl)) and zl < (-(zinl-50*dx)) ):
      xinbi = int(xinb/dx); zinbi = int(zinb/dz+0.5)
      #if(xinbi >= 0 and xinbi < nx and zinbi >= 0 and zinbi < nz):
      disp[ix,iz] = nimg[xinbi,zinbi]
      bini[ix,iz] =  1.0
    if( zl < (-(zinl)) and zl > (-(zinl+50*dx)) ):
      bini[ix,iz] = -1.0
      disp[ix,iz] = nimg[xinbi,zinbi]

plt.figure(7)
plt.imshow(bini,cmap='jet')

plt.figure(8)
plt.imshow(disp,cmap='jet')

plt.show()

