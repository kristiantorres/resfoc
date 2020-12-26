import inpout.seppy as seppy
import numpy as np
from seis.f3utils import plot_acq

sep = seppy.sep()

# Read in geometry
sxaxes,srcx = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_srcx3_full.H")
syaxes,srcy = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_srcy3_full.H")
rxaxes,recx = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_recx3_full.H")
ryaxes,recy = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_recy3_full.H")

naxes,nrec = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_nrec3_full.H")
nrec = nrec.astype('int32')

srcx *= 0.001
srcy *= 0.001
recx *= 0.001
recy *= 0.001

# Read in time slice for QC
saxes,slc = sep.read_wind("migwt.T",fw=400,nw=1)
dy,dx,dt = saxes.d; oy,ox,ot = saxes.o
slc = slc.reshape(saxes.n,order='F')
slcw = slc[25:125,:500]

plot_acq(srcx,srcy,recx,recy,slc,ox=ox,oy=oy,recs=False,show=True)


