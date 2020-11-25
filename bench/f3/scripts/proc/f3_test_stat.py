import inpout.seppy as seppy
import numpy as np
from pef.stat.pef1d import gapped_pef
from pef.stat.conv1dm import conv1dm
from oway.mute import mute
from genutils.plot import plot_dat2d, plot_img2d
import matplotlib.pyplot as plt

sep = seppy.sep()
saxes,sht = sep.read_wind("/data3/northsea_dutch_f3/f3_shots2.H",fw=0,nw=120)
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
deb = np.zeros(sht.shape,dtype='float32')
ntr,nt = sht.shape
dt = 0.002

smute = np.squeeze(mute(sht,dt=dt,dx=0.025,v0=1.5,t0=0.2,half=False))

plot_dat2d(sht,pclip=0.02,dt=dt,aspect='auto',show=False)
plot_dat2d(smute,pclip=0.02,dt=dt,aspect='auto')

plt.figure(); plt.plot(sht[20]); plt.show()

lags,invflt = gapped_pef(smute[20],na=50,gap=20,niter=300,verb=False)
cop = conv1dm(nt,len(lags),lags,flt=invflt)

for itr in range(ntr):
  cop.forward(False,smute[itr],deb[itr])

plot_dat2d(deb ,pclip=0.02,dt=dt,aspect='auto',show=False)
plot_dat2d(smute,pclip=0.02,dt=dt,aspect='auto')

#sep.write_file("f3_mute.H",smute.T,ds=[0.002,1.0])
#sep.write_file("f3_debub.H",deb.T,ds=[0.002,1.0])

