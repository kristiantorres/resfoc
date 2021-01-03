import numpy as np
import inpout.seppy as seppy
from seis.f3utils import select_f3shotcont, mute_f3shot
from genutils.plot import plot_dat2d

# Geometry
sep = seppy.sep()
sxaxes,srcx = sep.read_file("f3_srcx3_full_clean2.H")
syaxes,srcy = sep.read_file("f3_srcy3_full_clean2.H")
rxaxes,recx = sep.read_file("f3_recx3_full_clean2.H")
ryaxes,recy = sep.read_file("f3_recy3_full_clean2.H")
naxes,nrec = sep.read_file("f3_nrec3_full_clean2.H")
saxes,strm = sep.read_file("f3_strm3_full_clean2.H")
nrec = nrec.astype('int32')

dt = 0.004
# Read in the shots of interest
hdr1,dat1 = select_f3shotcont("f3_shots3interp_full_clean3.H",srcy,srcx,recy,recx,nrec,sx=485701,sy=6083753)

strm = np.asarray(list(range(1,115)) + list(range(1,121)))
hdr1['recx'][114] = hdr1['recx'][120]
hdr1['recy'][114] = hdr1['recy'][120]
mute = mute_f3shot(dat1,485701,6083753,dat1.shape[0],strm,hdr1['recx'],hdr1['recy'],dt=dt)

plot_dat2d(dat1,pclip=0.02,dt=dt,aspect=50,show=False)
plot_dat2d(mute,pclip=0.05,dt=dt,aspect=50)


