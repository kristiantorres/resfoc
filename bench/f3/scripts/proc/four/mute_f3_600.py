import inpout.seppy as seppy
import numpy as np
from seis.f3utils import mute_f3shot
from genutils.plot import plot_dat2d, plot_img2d
from genutils.ptyprint import progressbar
import matplotlib.pyplot as plt

# Geometry
sep = seppy.sep()
sxaxes,srcx = sep.read_file("/data3/northsea_dutch_f3/f3_srcx2_600.H")
syaxes,srcy = sep.read_file("/data3/northsea_dutch_f3/f3_srcy2_600.H")
rxaxes,recx = sep.read_file("/data3/northsea_dutch_f3/f3_recx2_600.H")
ryaxes,recy = sep.read_file("/data3/northsea_dutch_f3/f3_recy2_600.H")
naxes,nrec = sep.read_file("/data3/northsea_dutch_f3/f3_nrec2_600.H")
nrec = nrec.astype('int32')
nsht = 1950
nd = np.sum(nrec[:nsht])

# Data
saxes,sht = sep.read_wind("/data3/northsea_dutch_f3/f3_shots2_600.H",fw=0,nw=nd)
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
smute = np.zeros(sht.shape,dtype='float32')
ntr,nt = sht.shape
dt = 0.002
dmin,dmax = np.min(sht),np.max(sht)

idxs = [1340,1383]

ntr = 0
for isht in progressbar(range(nsht),"nsht",verb=True):
  if(isht in idxs):
    dymin = 3.0
    smute[ntr:] = mute_f3shot(sht[ntr:],srcx[isht],srcy[isht],nrec[isht],recx[ntr:],recy[ntr:],dymin=dymin)
    #plot_dat2d(sht[ntr:ntr+nrec[isht],:1500],show=False,dt=dt,dmin=dmin,dmax=dmax,pclip=0.01,aspect=50)
    #plot_dat2d(smute[ntr:ntr+nrec[isht],:1500],dt=dt,dmin=dmin,dmax=dmax,pclip=0.01,aspect=50)
  else:
    dymin = 15.0
    smute[ntr:] = mute_f3shot(sht[ntr:],srcx[isht],srcy[isht],nrec[isht],recx[ntr:],recy[ntr:],dymin=dymin)
  ntr += nrec[isht]

sep.write_file("f3_shots2_600_muted.H",smute.T,ds=[dt,1.0])

