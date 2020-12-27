import inpout.seppy as seppy
import numpy as np
from seis.f3utils import mute_f3shot
from genutils.plot import plot_dat2d, plot_img2d
from genutils.ptyprint import progressbar
import matplotlib.pyplot as plt

# Geometry
sep = seppy.sep()
sxaxes,srcx = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_srcx2.H")
syaxes,srcy = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_srcy2.H")
rxaxes,recx = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_recx2.H")
ryaxes,recy = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_recy2.H")
naxes,nrec = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_nrec2.H")
saxes,strm = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_strm2.H")
nrec = nrec.astype('int32')
nsht = 1625
nd = np.sum(nrec[:nsht])

# Data
saxes,sht = sep.read_wind("f3_shots2interp.H",fw=0,nw=nd)
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
smute = np.zeros(sht.shape,dtype='float32')
ntr,nt = sht.shape
dt = 0.004
dmin,dmax = np.min(sht),np.max(sht)

ntr = 0
for isht in progressbar(range(nsht),"nsht",verb=True):
  smute[ntr:] = mute_f3shot(sht[ntr:],srcx[isht],srcy[isht],nrec[isht],strm[ntr:],recx[ntr:],recy[ntr:],dt=dt,tp=0.25,hyper=False)
  #plot_dat2d(sht[ntr:ntr+nrec[isht],:700],show=False,dt=dt,dmin=dmin,dmax=dmax,pclip=0.01,aspect=50)
  #plot_dat2d(smute[ntr:ntr+nrec[isht],:700],dt=dt,dmin=dmin,dmax=dmax,pclip=0.01,aspect=50)
  ntr += nrec[isht]

sep.write_file("f3_shots2interp_muteda.H",smute.T,ds=[dt,1.0])

