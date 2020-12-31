import inpout.seppy as seppy
import numpy as np
from oway.mute import mute
from genutils.plot import plot_dat2d, plot_img2d
from genutils.movie import qc_f3data
import matplotlib.pyplot as plt

# Data
sep = seppy.sep()
#saxes,sht = sep.read_file("f3_shots2interp_muteda_debub_onetr.H")
#saxes,sht = sep.read_file("f3_shots2interp_muted_debub_onetr.H")
#saxes,sht = sep.read_file("f3_shots2interp_full_muted.H")
#saxes,sht = sep.read_file("f3_shots2interp_full_muted_debub_onetr.H")
#saxes,sht = sep.read_file("f3_shots3interp_full_clean.H")
saxes,sht = sep.read_file("f3_shots3interp_full_muted.H")
#saxes,sht = sep.read_wind("f3_shots3interp_full_muted.H",fw=0,nw=1000000)
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
ntr,nt = sht.shape
dt = 0.004

# Geometry
sxaxes,srcx = sep.read_file("f3_srcx3_full_clean.H")
syaxes,srcy = sep.read_file("f3_srcy3_full_clean.H")
rxaxes,recx = sep.read_file("f3_recx3_full_clean.H")
ryaxes,recy = sep.read_file("f3_recy3_full_clean.H")

naxes,nrec = sep.read_file("f3_nrec3_full_clean.H")
nrec = nrec.astype('int32')
nsht = 12000
nd = np.sum(nrec[:nsht])

# Read in time slice for QC
saxes,slc = sep.read_wind("migwt.T",fw=450,nw=1)
dy,dx,dt = saxes.d; oy,ox,ot = saxes.o
slc = slc.reshape(saxes.n,order='F').T

qc_f3data(sht[:nd],srcx[:nsht],recx[:nd],srcy[:nsht],recy[:nd],nrec[:nsht],slc,dt=0.004,
          pclip=0.02,ntw=750,sjump=125)

