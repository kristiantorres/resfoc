import inpout.seppy as seppy
import numpy as np
from oway.mute import mute
from genutils.plot import plot_dat2d, plot_img2d
from genutils.movie import qc_f3data
import matplotlib.pyplot as plt

# Data
sep = seppy.sep()
saxes,sht = sep.read_file("/data3/northsea_dutch_f3/f3_shots2_full.H")
#saxes,sht = sep.read_file("f3_shots2_full_muted.H")
#saxes,sht = sep.read_file("f3_shots2_muted.H")
#saxes,sht = sep.read_file("f3_shots2_muted_debub_onetr.H")
#saxes,sht = sep.read_file("f3_shots2_muted_debub_shot.H")
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
ntr,nt = sht.shape
dt = 0.002

# Geometry
sxaxes,srcx = sep.read_file("/data3/northsea_dutch_f3/f3_srcx2_full.H")
syaxes,srcy = sep.read_file("/data3/northsea_dutch_f3/f3_srcy2_full.H")
rxaxes,recx = sep.read_file("/data3/northsea_dutch_f3/f3_recx2_full.H")
ryaxes,recy = sep.read_file("/data3/northsea_dutch_f3/f3_recy2_full.H")

naxes,nrec = sep.read_file("/data3/northsea_dutch_f3/f3_nrec2_full.H")
nrec = nrec.astype('int32')
nsht = 3250
nd = np.sum(nrec[:nsht])

# Read in time slice for QC
saxes,slc = sep.read_wind("migwt.T",fw=400,nw=1)
dy,dx,dt = saxes.d; oy,ox,ot = saxes.o
slc = slc.reshape(saxes.n,order='F').T

qc_f3data(sht[:nd],srcx[:nsht],recx[:nd],srcy[:nsht],recy[:nd],nrec[:nsht],slc,pclip=0.02)

