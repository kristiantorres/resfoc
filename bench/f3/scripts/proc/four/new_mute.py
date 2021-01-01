import inpout.seppy as seppy
import numpy as np
from genutils.plot import plot_dat2d

# Read in one shot data and headers
sep = seppy.sep()
sxaxes,srcx = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_srcx2_full.H")
syaxes,srcy = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_srcy2_full.H")
rxaxes,recx = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_recx2_full.H")
ryaxes,recy = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_recy2_full.H")
staxes,strm = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_strm2_full.H")
naxes,nrec = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_nrec2_full.H")
nrec = nrec.astype('int32')
nsht = 1
nd = np.sum(nrec[:nsht])

# Data
saxes,sht = sep.read_wind("f3_shots2interp_full.H",fw=0,nw=nd)
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
smute = np.zeros(sht.shape,dtype='float32')
ntr,nt = sht.shape
dt = 0.004
dmin,dmax = np.min(sht),np.max(sht)

# Find each streamer line
strmw = strm[:480]
idxs = np.where(strmw == 1)[0]
print(idxs)

for istrm in range(1,len(idxs)):
  line = sht[idxs[istrm-1]:idxs[istrm]]
  print(recx[idxs[istrm-1]],recy[idxs[istrm-1]])
  plot_dat2d(line,dt=dt,pclip=0.02,aspect=100)


