import numpy as np
from scaas.noise_generator import perlin
import matplotlib.pyplot as plt

nx = 1000; ny = 100; nz = 100;
nptsx = 2; nptsy = 3; nptsz = 3

# Default arguments
octave = 4; period = 10.0; amp=1.0; persist=0.5; Ngrad = 80

# 1D plots
# Plotting octaves
nocts = 8
octs1 = np.zeros([nocts,nx])
fig = plt.figure(1,figsize=(10,7)); ax = fig.gca()
for ioct in range(1,nocts):
  octs1[ioct] = perlin(x=np.linspace(0,nptsx,nx),octaves=ioct,amp=amp,persist=persist,Ngrad=Ngrad,period=period)
  lin, = ax.plot(octs1[ioct])
  lin.set_label('Octave=%d'%(ioct))
ax.legend(prop={"size":12})
ax.tick_params(labelsize=14)

# Plotting persist
npsts = 5; dpst = 0.2; opst = 0.1
psts1 = np.zeros([npsts,nx])
fig = plt.figure(2,figsize=(10,7)); ax = fig.gca()
for ipst in range(npsts):
  pst = opst + dpst*ipst
  psts1[ipst] = perlin(x=np.linspace(0,nptsx,nx),octaves=octave,amp=amp,persist=pst,Ngrad=Ngrad,period=period)
  lin, = ax.plot(psts1[ipst])
  lin.set_label('Persist=%.2f'%(pst))
ax.legend(prop={"size":12})
ax.tick_params(labelsize=14)

# Plotting npts
nnpts = 10
pts1 = np.zeros([nnpts,nx])
fig = plt.figure(3,figsize=(10,7)); ax = fig.gca()
for ipts in range(1,nnpts,2):
  pts1[ipts] = perlin(x=np.linspace(0,ipts,nx),octaves=2,amp=amp,persist=persist,Ngrad=Ngrad,period=period)
  lin, = ax.plot(pts1[ipts])
  lin.set_label('Npoints=%d'%(ipts))
ax.legend(prop={"size":12})
ax.tick_params(labelsize=14)

# Plotting periods
fig = plt.figure(4,figsize=(10,7)); ax = fig.gca()
per1 = perlin(x=np.linspace(0,nptsx,nx),octaves=4,amp=amp,persist=persist,Ngrad=Ngrad,period=0.5)
lin1, = ax.plot(per1)
lin1.set_label("Period=0.5,Npts=2")
# Keep same period but increase ptsx
per2 = perlin(x=np.linspace(0,nptsx*2,nx),octaves=4,amp=amp,persist=persist,Ngrad=Ngrad,period=0.5)
lin2, = ax.plot(per2)
lin2.set_label("Period=0.5,Npts=4")
ax.legend(prop={"size":12})
ax.tick_params(labelsize=14)

plt.show()

