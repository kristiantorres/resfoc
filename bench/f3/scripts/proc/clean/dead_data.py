import numpy as np
from seis.f3utils import select_f3shot
from genutils.plot import plot_dat2d

sx,sy = 486194,6080750
hdrkeys = ['TraceIdentificationCode']
hdr,dat = select_f3shot(sx,sy,hdrkeys=hdrkeys)

print(hdr['TraceIdentificationCode'])

plot_dat2d(dat,aspect='auto',dt=0.002,pclip=0.02,title='Dead shot',show=True)
