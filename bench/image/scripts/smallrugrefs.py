import numpy as np
import inpout.seppy as seppy
from scaas.velocity import create_layered
from scaas.trismooth import smooth
import matplotlib.pyplot as plt

# Model dimensions
nx = 256; dx = 10
nz = 256; dz = 10

# Create the rugose layered model
ovel,lyrs = create_layered(nz,nx,dz,dx,z0s=[30,70,120,200,250],vels=[1500,1800,2500,2800,3200,3400],flat=False,scale=150)

# Smooth the interfaces
velsm = smooth(ovel.astype('float32'),rect1=2,rect2=2)

plt.figure()
plt.imshow(ovel,cmap='jet')

plt.figure()
plt.imshow(velsm,cmap='jet')

plt.figure()
plt.imshow(lyrs,cmap='gray',vmin=-1,vmax=1)
plt.show()

# Write the smoothed velocity and the reflectivity
sep = seppy.sep()
sep.write_file("velrug.H",velsm,ds=[10,10])
sep.write_file("lyrrug.H",lyrs,ds=[10,10])
