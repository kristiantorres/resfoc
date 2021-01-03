import inpout.seppy as seppy
import numpy as np
from oway.mute import mute
from genutils.plot import plot_dat2d, plot_img2d
from genutils.movie import qc_f3data
import matplotlib.pyplot as plt

# Data
sep = seppy.sep()

# Geometry
fsht,nsht = 2000,3000
sxaxes,srcx = sep.read_wind("f3_srcx3_full_clean2.H",fw=fsht,nw=nsht)
syaxes,srcy = sep.read_wind("f3_srcy3_full_clean2.H",fw=fsht,nw=nsht)

naxes,nrec = sep.read_file("f3_nrec3_full_clean2.H")
nrec = nrec.astype('int32')

bred = np.sum(nrec[:fsht])
nred = np.sum(nrec[fsht:fsht+nsht])
rxaxes,recx = sep.read_wind("f3_recx3_full_clean2.H",fw=bred,nw=nred)
ryaxes,recy = sep.read_wind("f3_recy3_full_clean2.H",fw=bred,nw=nred)

saxes,sht = sep.read_wind("f3_shots3interp_full_muted_debub_onetr.H",fw=bred,nw=nred)
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
ntr,nt = sht.shape
dt = 0.004

# Read in time slice for QC
saxes,slc = sep.read_wind("migwt.T",fw=450,nw=1)
dy,dx,dt = saxes.d; oy,ox,ot = saxes.o
slc = slc.reshape(saxes.n,order='F').T

qc_f3data(sht,srcx,recx,srcy,recy,nrec[fsht:fsht+nsht],slc,dt=dt,pclip=0.02,ntw=750,sjump=10)

