import numpy as np
from scaas.noise_generator import perlin
import matplotlib.pyplot as plt
from utils.plot import plot3d

nx = 100; ny = 100; nz = 100
nptsx = 2; nptsy = 3; nptsz = 2

# Default arguments
octave = 4; period = 10.0; amp=1.0; persist=0.5; Ngrad = 80; ncpu = 1

## 3D plots
# Plotting octaves
octs = np.zeros([nz,ny,nx])
octs = perlin(x=np.linspace(0,nptsx,nx),y=np.linspace(0,nptsy,ny),z=np.linspace(0,nptsz,nz),
              octaves=octave,amp=amp,persist=persist,Ngrad=Ngrad,period=period,ncpu=ncpu)


plot3d(octs)
