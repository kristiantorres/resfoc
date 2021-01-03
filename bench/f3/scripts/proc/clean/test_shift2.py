import numpy as np
import inpout.seppy as seppy
from seis.f3utils import select_f3shotcont
from genutils.plot import plot_dat2d

def correct_shift(dat,shift1,shift2):
  odat = np.zeros(dat.shape,dtype='float32')
  odat[  0:120] = np.roll(dat[  0:120],shift1,axis=1)
  odat[120:240] = np.roll(dat[120:240],shift2,axis=1)
  odat[240:]    = dat[240:]
  return odat

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
hdr1,dat1 = select_f3shotcont("f3_shots3interp_full_clean2.H",srcy,srcx,recy,recx,nrec,sx=485297,sy=6082913)
hdr1a,dat1a = select_f3shotcont("f3_shots3interp_full_clean2.H",srcy,srcx,recy,recx,nrec,sx=485370,sy=6082910)

plot_dat2d(correct_shift(dat1,-20,-21),
           dt=dt,pclip=0.02,aspect=50,title='Shifted',show=False)
plot_dat2d(dat1a,dt=dt,pclip=0.02,aspect=50,title='Correct',show=True)

hdr2,dat2   = select_f3shotcont("f3_shots3interp_full_clean2.H",srcy,srcx,recy,recx,nrec,sx=485679,sy=6082902)
hdr2a,dat2a = select_f3shotcont("f3_shots3interp_full_clean2.H",srcy,srcx,recy,recx,nrec,sx=485599,sy=6082901)

plot_dat2d(correct_shift(dat2,-30,-31),
           dt=dt,pclip=0.02,aspect=50,title='Shifted',show=False)
plot_dat2d(dat2a,dt=dt,pclip=0.02,aspect=50,title='Correct',show=True)

hdr3,dat3   = select_f3shotcont("f3_shots3interp_full_clean2.H",srcy,srcx,recy,recx,nrec,sx=485785,sy=6076349)
hdr3a,dat3a = select_f3shotcont("f3_shots3interp_full_clean2.H",srcy,srcx,recy,recx,nrec,sx=485711,sy=6076348)

plot_dat2d(correct_shift(dat3,-30,-31),
           dt=dt,pclip=0.02,aspect=50,title='Shifted',show=False)
plot_dat2d(dat3a,dt=dt,pclip=0.02,aspect=50,title='Correct',show=True)

