import numpy as np
import inpout.seppy as seppy
import inpout.pv as pv
import matplotlib.pyplot as plt
from utils.movie import viewframeskey

# Create a meshgrid
nx = 128; ox = 0.0; dx = 10.0
nz = 128; oz = 0.0; dz = 10.0
ny = 128; oy = 0.0; dy = 10.0

x = np.linspace(ox,ox+(nx-1)*dx,nx)
z = np.linspace(oz,oz+(nz-1)*dz,nz)
y = np.linspace(oy,oy+(ny-1)*dy,ny)

X,Y,Z = np.meshgrid(x,y,z)

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
X0 = cx*dx; Y0 = cy*dy; Z0 = cz*dz

# Choose the strike and dip angles
phi = 90; theta = 45

# Convert to radians
phir = np.pi/180*phi; thetar = np.pi/180*theta

# Create local coordinate mesh
x = np.sin(phir)               *(X-X0) + np.cos(phir)               *(Y-Y0) + 0             *(Z-Z0)
y = np.cos(phir)*np.cos(thetar)*(X-X0) - np.sin(phir)*np.cos(thetar)*(Y-Y0) + np.sin(thetar)*(Z-Z0)
z = np.cos(phir)*np.sin(thetar)*(X-X0) + np.sin(phir)*np.sin(thetar)*(Y-Y0) - np.cos(thetar)*(Z-Z0)

# Assign a constant value at all x and y and z=0 in the new coordinate system
dx = x[0,2,0] - x[0,1,0]
dy = y[0,0,2] - y[0,0,1]
dz = z[2,0,0] - z[1,0,0]

# Find all indices where z == 0
zidx = np.abs(z) < 5.0

cub[zidx] = 1

print(np.min(np.abs(z)))

xidx = x/dx
yidx = y/dy

sep = seppy.sep([])
axes = seppy.axes([ny,nx,nz],[0.0,0.0,0.0],[1.0,1.0,1.0])
sep.write_file(None,axes,Z,"globz.H")
sep.write_file(None,axes,z,"loclz.H")

sep.write_file(None,axes,cub,"testplane.H")

pv.write_nrrd('z60.nrrd',z)
pv.write_nrrd('plane.nrrd',cub,ds=[dy,dx,dx])

viewframeskey(cub,cmap='jet')

