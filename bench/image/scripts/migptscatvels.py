import inpout.seppy as seppy
import numpy as np
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
import matplotlib.pyplot as plt
from genutils.plot import plot_wavelet
from genutils.movie import viewimgframeskey

# Create point scatterer model
nz  = 251; oz  =  0.0;  dz  = 10.0
nx  = 501 ; ox  =  0.0;  dx  = 10.0

# Make a spike
pt = np.zeros([nz,nx],dtype='float32')
pt[125,250] = 1.0

# Velocity model
vel = np.zeros([nz,nx],dtype='float32')
vel[:] = 2500.0

dsx = 10
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=52,dsx=dsx,bx=50,bz=50)
prp.plot_acq(vel)

# Data axes
ntu=2500; dtu = 0.001
freq = 20; amp = 100.0; dly = 0.1
wav = ricker(ntu,dtu,freq,amp,dly)
plot_wavelet(wav,dtu)

dtd = 0.004
# Model the data
allshot = prp.model_lindata(vel,pt,wav,dtd,verb=True,nthrds=24)

#viewimgframeskey(allshot,transp=False,zmax=dtd*(2500/4))

# Migration
percs = [1.2,1.1,1.05,1.0,0.95,0.9,0.8]; imgs = np.zeros([len(percs),nz,nx],dtype='float32')
k = 0
for iperc in percs:
  print("Percent error: %.2f"%(iperc))
  imgs[k] = prp.wem(vel*iperc,allshot,wav,dtd,verb=True,nthrds=24)
  k += 1

sep = seppy.sep()
sep.write_file("ptscatvels.H",imgs.T,ds=[dz,dx,1])


