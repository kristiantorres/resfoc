import inpout.seppy as seppy
import numpy as np
from pef.nstat.peflms1d import peflmsgap1d
from oway.mute import mute
from genutils.plot import plot_dat2d, plot_img2d
import matplotlib.pyplot as plt

sep = seppy.sep()
saxes,sht = sep.read_wind("/data3/northsea_dutch_f3/f3_shots.H",fw=0,nw=120)
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
deb = np.zeros(sht.shape,dtype='float32')
ntr,nt = sht.shape
dt = 0.002

smute = np.squeeze(mute(sht,dt=dt,dx=0.025,v0=1.5,t0=0.2,half=False))

plot_dat2d(sht,pclip=0.01,dt=dt,aspect='auto',show=False)
plot_dat2d(smute,pclip=0.01,dt=dt,aspect='auto')

plot_dat2d(sht,pclip=0.01,aspect='auto',show=False)
nw = 50
a = np.zeros([nw],dtype='float32')

flts = np.zeros([ntr,nw],dtype='float32')

for itr in range(ntr):
  err,a = peflmsgap1d(smute[itr],nw=nw,gap=20,mu=0.005,w0=a,update=True)
  flts[itr,:] = a[:]
  deb[itr,:] = err[:]

plot_img2d(flts.T,show=True,aspect='auto',interp='none')

plot_dat2d(deb,pclip=0.01,aspect='auto')

sep.write_file("f3_mute.H",smute.T,ds=[0.002,1.0])
sep.write_file("f3_debub.H",deb.T,ds=[0.002,1.0])

