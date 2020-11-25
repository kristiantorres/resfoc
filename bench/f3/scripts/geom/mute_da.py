import inpout.seppy as seppy
import numpy as np
from oway.mute import mute
from genutils.signal import ampspec2d
from genutils.plot import plot_dat2d
import matplotlib.pyplot as plt

#TODO: check the receiver and temporal sampling
dt = 0.002; dtr = 0.025

sep = seppy.sep()
# Read in data
daxes,dat = sep.read_wind("f3_shots.H",nw=480,fw=0)
dt,dtr = daxes.d; ot,otr = daxes.o
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
print(dat.shape)

# Read in source coordinates
_,srcx = sep.read_wind("f3_srcx.H",nw=1,fw=0)
_,srcy = sep.read_wind("f3_srcy.H",nw=1,fw=0)

# Read in receiver coordinates
_,recx = sep.read_wind("f3_recx.H",nw=480,fw=0)
_,recy = sep.read_wind("f3_recy.H",nw=480,fw=0)

# Read in number of receivers per shot
_,nrec = sep.read_wind("f3_nrec.H",nw=1,fw=0)

plot_dat2d(dat,pclip=0.01,aspect='auto')

datfft,k1,k2 = ampspec2d(dat[:120,:],dt,dtr)

fig = plt.figure(); ax = fig.gca()
ax.imshow(datfft,extent=[k1[0],k1[-1],k2[0],k2[-1]],interpolation='bilinear',aspect='auto')
plt.show()
