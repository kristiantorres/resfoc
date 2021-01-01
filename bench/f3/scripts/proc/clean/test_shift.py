import numpy as np
from seis.f3utils import select_f3shot
from genutils.plot import plot_dat2d

def correct_shift(dat,shift1,shift2):
  odat = np.zeros(dat.shape,dtype='float32')
  odat[  0:120] = np.roll(dat[  0:120],shift1,axis=1)
  odat[120:240] = np.roll(dat[120:240],shift2,axis=1)
  odat[240:]    = dat[240:]
  return odat

dt = 0.002
# Read in the shots of interest
hdr1,dat1   = select_f3shot(sx=485679,sy=6082902)
hdr1a,dat1a = select_f3shot(sx=485370,sy=6082910)

plot_dat2d(correct_shift(dat1,-60,-62),
           dt=dt,pclip=0.02,aspect=50,title='Shifted',show=False)
plot_dat2d(dat1a,dt=dt,pclip=0.02,aspect=50,title='Correct',show=True)

hdr2,dat2   = select_f3shot(sx=485679,sy=6082902)
hdr2a,dat2a = select_f3shot(sx=485599,sy=6082901)

plot_dat2d(correct_shift(dat2,-60,-62),
           dt=dt,pclip=0.02,aspect=50,title='Shifted',show=False)
plot_dat2d(dat2a,dt=dt,pclip=0.02,aspect=50,title='Correct',show=True)

hdr3,dat3   = select_f3shot(sx=485785,sy=6076349)
hdr3a,dat3a = select_f3shot(sx=485711,sy=6076348)

plot_dat2d(correct_shift(dat3,-60,-62),
           dt=dt,pclip=0.02,aspect=50,title='Shifted',show=False)
plot_dat2d(dat3,dt=dt,pclip=0.02,aspect=50,title='Correct',show=True)
