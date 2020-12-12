import inpout.seppy as seppy
import numpy as np
from genutils.plot import plot_dat2d

sep = seppy.sep()

daxes,dat = sep.read_file("f3_shots2_muted_debub_onetr.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')

datp = np.pad(dat,((0,0),(0,4)))

# Shift by 8 milliseconds
datpm = np.roll(datp,4,axis=1)

# Window to 6 s
datpmw = datpm[:,:3000]

#plot_dat2d(dat[:1000],dt=0.002,pclip=0.1,show=False,aspect=60.0)
#plot_dat2d(datpm[:1000],dt=0.002,pclip=0.1,show=True,aspect=60.0)

sep.write_file("f3_shots2_muted_debub_onetr_gncw.H",datpmw.T,ds=daxes.d,os=daxes.o)
