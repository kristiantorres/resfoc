import inpout.seppy as seppy
import numpy as np
from genutils.ptyprint import progressbar
from genutils.plot import plot_dat2d

def correct_shift(dat,shift1,shift2):
  odat = np.zeros(dat.shape,dtype='float32')
  odat[  0:120] = np.roll(dat[  0:120],shift1,axis=1)
  odat[120:240] = np.roll(dat[120:240],shift2,axis=1)
  odat[240:]    = dat[240:]
  return odat

sep = seppy.sep()
# Read in the geometry
sxaxes,srcx = sep.read_file("f3_srcx3_full_clean2.H")
syaxes,srcy = sep.read_file("f3_srcy3_full_clean2.H")
rxaxes,recx = sep.read_file("f3_recx3_full_clean2.H")
ryaxes,recy = sep.read_file("f3_recy3_full_clean2.H")
naxes,nrec  = sep.read_file("f3_nrec3_full_clean2.H")
saxes,strm  = sep.read_file("f3_strm3_full_clean2.H")
nrec = nrec.astype('int32')

# Read in the data
daxes,dat = sep.read_file("f3_shots3interp_full_clean2.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')

idxs = set([4835,12805,12809])
lsts = [[-30,31],[],[]]

ntr = 0
for isht in progressbar(range(len(srcx)),"nshots:"):
  #plot_dat2d(dat[ntr:ntr+nrec[isht]],dt=0.004,pclip=0.02,aspect=50,show=True)
  if(isht in idxs):
    plot_dat2d(dat[ntr:ntr+nrec[isht]],dt=0.004,pclip=0.02,title='Shifted',aspect=50,show=False)
    if(isht == 12805):
      dat[ntr:ntr+nrec[isht]] = correct_shift(dat[ntr:ntr+nrec[isht]],-20,-21)
    else:
      dat[ntr:ntr+nrec[isht]] = correct_shift(dat[ntr:ntr+nrec[isht]],-30,-31)
    plot_dat2d(dat[ntr:ntr+nrec[isht]],dt=0.004,pclip=0.02,title='Correct',aspect=50,show=True)
  ntr += nrec[isht]

sep.write_file("f3_shots3interp_full_clean3.H",dat.T,ds=[0.004,1.0])

