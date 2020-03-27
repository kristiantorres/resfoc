import numpy as np
import inpout.seppy as seppy
import inpout.pv as pv
import matplotlib.pyplot as plt
from utils.movie import viewframeskey

def dist(x,x0,y,y0,lx=500,ly=500):
  return np.sqrt( ((x-x0)/lx)**2 + ((y-y0)/ly)**2 )

# Create a meshgrid
nx = 128; ox = 0.0; dx = 10.0
nz = 128; oz = 0.0; dz = 10.0
ny = 128; oy = 0.0; dy = 10.0

x = np.linspace(ox,ox+(nx-1)*dx,nx)
z = np.linspace(oz,oz+(nz-1)*dz,nz)
y = np.linspace(oy,oy+(ny-1)*dy,ny)

# Create a 3D cube
cub = np.zeros([ny,nx,nz],dtype='float32')

# Create an arbitrary plane through the cube
# The plane is defined with the following parameters
#  1. Origin (x,y,z) (like Bob, this can be computed as a percent distance of the model)
#  2. Dip direction (the dip is defined as the angle from horizontal)
#  3. Strike or azimuth (defined as the angle from x = 0)

# First create the center coordinate
cx = 63; cy = 63; cz = 63
#X0 = X[0,cx,0]; Y0 = Y[0,0,cy]; Z0 = Z[cz,0,0]
x0 = cx*dx; y0 = cy*dy; z0 = cz*dz

# Choose the strike and dip angles
phi = 90; theta = 45

# Convert to radians
phir = np.pi/180*phi; thetar = np.pi/180*theta

dmax = 10

# Loop over each point
for iy in range(ny):
  y = oy + iy*dy
  for ix in range(nx):
    x = ox + ix*dx
    for iz in range(nz):
      z = oz + iz*dz
      # Rotate the point into the local coordinate system
      xl = np.sin(phir)               *(x-x0) + np.cos(phir)               *(y-y0) + 0             *(z-z0)
      yl = np.cos(phir)*np.cos(thetar)*(x-x0) - np.sin(phir)*np.cos(thetar)*(y-y0) + np.sin(thetar)*(z-z0)
      zl = np.cos(phir)*np.sin(thetar)*(x-x0) + np.sin(phir)*np.sin(thetar)*(y-y0) + np.cos(thetar)*(z-z0)
      # Compute the elliptic function
      if(np.abs(zl) < 5):
        rxy = dist(xl,0.0,yl,0.0,lx=700,ly=1000)
        scale = ((1+rxy)**2)/4 - rxy**2
        if(scale >= 0):
          cub[iy,ix,iz] = 2*dmax*(1-rxy) * np.sqrt(scale)
        else:
          cub[iy,ix,iz] = 0.0

pv.write_nrrd('planenew.nrrd',cub,ds=[dy,dx,dz])


viewframeskey(cub,cmap='jet',scalebar=True,wbox=14,hbox=7)

