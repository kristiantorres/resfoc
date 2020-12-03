import inpout.seppy as seppy
import numpy as np
from oway.mute import mute
from genutils.plot import plot_dat2d, plot_img2d
from genutils.movie import qc_f3data
import matplotlib.pyplot as plt

# Data
sep = seppy.sep()
#saxes,sht = sep.read_file("/data3/northsea_dutch_f3/f3_shots2.H")
saxes,sht = sep.read_file("f3_shots2_muted.H")
#saxes,sht = sep.read_file("f3_shots2_muted_debub_onetr.H")
#saxes,sht = sep.read_file("f3_shots2_muted_debub_shot.H")
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
ntr,nt = sht.shape
dt = 0.002

# Geometry
sxaxes,srcx = sep.read_file("/data3/northsea_dutch_f3/f3_srcx2.H")
syaxes,srcy = sep.read_file("/data3/northsea_dutch_f3/f3_srcy2.H")
rxaxes,recx = sep.read_file("/data3/northsea_dutch_f3/f3_recx2.H")
ryaxes,recy = sep.read_file("/data3/northsea_dutch_f3/f3_recy2.H")

naxes,nrec = sep.read_file("/data3/northsea_dutch_f3/f3_nrec2.H")
nrec = nrec.astype('int32')
nsht = 1625
nd = np.sum(nrec[:nsht])

# Migration
maxes,mig = sep.read_file("/data3/northsea_dutch_f3/mig/mig.T")
mig = mig.reshape(maxes.n,order='F')
migw  = mig[:,200:1200,5:505]
migslc = migw[400]

qc_f3data(sht[:nd],srcx[:nsht],recx[:nd],srcy[:nsht],recy[:nd],nrec[:nsht],migslc,pclip=0.02)

