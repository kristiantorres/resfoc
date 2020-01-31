"""
Create an extended image of point scatterers
@author: Joseph Jennings
@version: 2019.12.18
"""
import sys, os
import inpout.seppy as seppy
import numpy as np
import scaas.scaas2dpy as sca2d
from scaas.wavelet import ricker
from scaas.velocity import create_ptscatmodel
import resfoc.cosft as cft
import resfoc.rstolt as rstolt
import matplotlib.pyplot as plt

## Create the point scatterer model
# Axes
nz = 200; oz = 0.0; dz = 20.0
nx = 400; ox = 0.0; dx = 20.0

# Background and point scatterer
velval = 2500.0
vel  = np.zeros([nz,nx],dtype='float32') + velval
dvel = create_ptscatmodel(nz,nx,50,50)

# Padding
bx = 50; bz = 50
velp  = np.pad(vel, ((bz,bz),(bx,bx)),'edge')
dvelp = np.pad(dvel,((bz,bz),(bx,bx)),'edge')

# Pad for laplacian
velp  = np.pad(velp, ((5,5),(5,5)),'edge')
dvelp = np.pad(dvelp,((5,5),(5,5)),'edge')

# Get padded lengths
nzp,nxp = velp.shape

## Model the born data
nsx = 41; osxp =  bx + 5; dsx = 10
srczp = bz + 5
nsrc = np.ones(nsx,dtype='int32')
allsrcx = np.zeros([nsx,1],dtype='int32')
allsrcz = np.zeros([nsx,1],dtype='int32')
# All source x positions in one array
srcs = np.linspace(osxp,osxp + (nsx-1)*dsx,nsx)
for isx in range(nsx):
  allsrcx[isx,0] = int(srcs[isx])
  allsrcz[isx,0] = int(srczp)

# Create receiver coordinates
nrx = nx; orxp = bx + 5; drx = 1
reczp = bz + 5
nrec = np.zeros(nrx,dtype='int32') + nrx
allrecx = np.zeros([nsx,nrx],dtype='int32')
allrecz = np.zeros([nsx,nrx],dtype='int32')
# Create all receiver positions
recs = np.linspace(orxp,orxp + (nrx-1)*drx,nrx)
for isx in range(nsx):
  allrecx[isx,:] = (recs[:]).astype('int32')
  allrecz[isx,:] = np.zeros(len(recs),dtype='int32') + reczp

# Plot acquisition
plt.figure(1)
# Plot velocity model
vtot = vel + dvel
vtotp = velp + dvelp
vmin = np.min(vtot); vmax = np.max(vtot)
plt.imshow(vtotp,extent=[0,nxp,nzp,0],vmin=vmin,vmax=vmax,cmap='jet')
# Get all source positions
plt.scatter(allrecx[0,:],allrecz[0,:])
plt.scatter(allsrcx[:,0],allsrcz[:,0])
plt.show()

# Create data axes
ntu = 4000; otu = 0.0; dtu = 0.001;
ntd = 1000; otd = 0.0; dtd = 0.004;

# Create the wavelet array
freq = 15; amp = 100.0; dly = 0.2;
fsrc = ricker(ntu,dtu,freq,amp,dly)
allsrcs = np.zeros([nsx,1,ntu],dtype='float32')
for isx in range(nsx):
  allsrcs[isx,0,:] = fsrc[:]

# Create output data array
fact = int(dtd/dtu);
ddat = np.zeros((nsx,ntd,nrx),dtype='float32')

# Set up a wave propagation object
sca = sca2d.scaas2d(ntd,nxp,nzp,dtd,dx,dz,dtu,bx,bz,alpha=0.99)

# Forward modeling for all shots
nthreads = 24
sca.brnfwd(allsrcs,allsrcx,allsrcz,nsrc,allrecx,allrecz,nrec,nsx,velp,dvelp,ddat,nthreads)

zidx = 10
rnh = 2*zidx + 1
imgp  = np.zeros([rnh,nzp,nxp],dtype='float32')
sca.brnoffadj(allsrcs,allsrcx,allsrcz,nsrc,allrecx,allrecz,nrec,nsx,velp,rnh,imgp,ddat,nthreads)

# Output extended image
imge = imgp[:,bz+5:nz+bz+5,bx+5:nx+bx+5]
# Make z fast axis
imget = np.ascontiguousarray(np.transpose(imge,(0,2,1)))
iaxes = seppy.axes([rnh,nx,nz],[-dx*zidx,0.0,0.0],[dx,dx,dz])
sep = seppy.sep([])
sep.write_file(None,iaxes,imget,ofname='eimg.H')

