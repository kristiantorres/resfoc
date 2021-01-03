import numpy as np
from seis.f3utils import select_f3shot, mute_f3shot
from genutils.plot import plot_dat2d

dt = 0.002
# Read in the shots of interest
hdr1,dat1   = select_f3shot(sx=485701,sy=6083753,hdrkeys=['CDP_TRACE','GroupX','GroupY'])

print(hdr1['GroupX'])

strm = np.asarray(list(range(1,115)) + list(range(1,121)))
hdr1['GroupX'][114] = hdr1['GroupX'][120]
hdr1['GroupY'][114] = hdr1['GroupY'][120]
mute = mute_f3shot(dat1,485701,6083753,dat1.shape[0],strm,hdr1['GroupX'],hdr1['GroupY'],dt=dt)

plot_dat2d(dat1,pclip=0.02,dt=dt,aspect=50,show=False)
plot_dat2d(mute,pclip=0.05,dt=dt,aspect=50)


