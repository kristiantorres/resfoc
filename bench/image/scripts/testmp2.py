from scaas.noise_generator import perlin
import numpy as np

nx = 100; ny = 100; nz = 100 
nptsx = 2; nptsy = 3; nptsz = 2 

# Default arguments
octave = 8; period = 10.0; amp=1.0; persist=0.5; ncpu = 8

# 3D plots
octs = np.zeros([nz,ny,nx])
octs = perlin(x=np.linspace(0,nptsx,nx),y=np.linspace(0,nptsy,ny),z=np.linspace(0,nptsz,nz),
              octaves=octave,amp=amp,persist=persist,period=period,ncpu=ncpu)
