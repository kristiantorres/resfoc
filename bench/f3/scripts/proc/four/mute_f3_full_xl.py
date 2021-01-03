import inpout.seppy as seppy
import numpy as np
from seis.f3utils import mute_f3shot, compute_batches_var
from genutils.plot import plot_dat2d
from genutils.ptyprint import progressbar
import matplotlib.pyplot as plt

# Geometry
sep = seppy.sep()
sxaxes,srcx = sep.read_file("f3_srcx3_full_clean2.H")
syaxes,srcy = sep.read_file("f3_srcy3_full_clean2.H")
rxaxes,recx = sep.read_file("f3_recx3_full_clean2.H")
ryaxes,recy = sep.read_file("f3_recy3_full_clean2.H")
naxes,nrec = sep.read_file("f3_nrec3_full_clean2.H")
saxes,strm = sep.read_file("f3_strm3_full_clean2.H")
nrec = nrec.astype('int32')
nsht = 13970
nd = np.sum(nrec[:nsht])

# Batch size for processing data
ibatch = 3000
bsizes = compute_batches_var(ibatch,nsht)
nb = len(bsizes)
print("Shot batch sizes: ",*bsizes)

# Read in the data in batches
totred,isht = 0,0
for ibtch in range(nb):
  print("Batch %d"%(ibtch))
  # Window the source geometry
  srcxw = srcx[isht:isht+bsizes[ibtch]]
  srcyw = srcy[isht:isht+bsizes[ibtch]]
  nrecw = nrec[isht:isht+bsizes[ibtch]]
  # Compute the number of traces to read in
  nred = np.sum(nrecw)
  # Window the receivers
  recxw = recx[totred:totred+nred]
  recyw = recy[totred:totred+nred]
  strmw = strm[totred:totred+nred]
  # Read in the data
  daxes,dat = sep.read_wind("f3_shots3interp_full_clean3.H",fw=totred,nw=nred)
  dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
  nt,ntr = daxes.n; ot,_ = daxes.o; dt,_ = daxes.d
  isht   += bsizes[ibtch]
  totred += nred
  dmin,dmax = np.min(dat),np.max(dat)

  # Output array
  smute = np.zeros(dat.shape,dtype='float32')

  # Residual da - 458
  ntrw = 0
  for iexp in progressbar(range(bsizes[ibtch]),"nsht",verb=True):
    if(srcxw[iexp] == 485701.0 and srcyw[iexp] == 6083753.0):
      strm = np.asarray(list(range(1,115)) + list(range(1,121)))
      recxw[ntrw+114] = recxw[ntrw+120]
      recyw[ntrw+114] = recyw[ntrw+120]
      smute[ntrw:] = mute_f3shot(dat[ntrw:],srcxw[iexp],srcyw[iexp],nrecw[iexp],strm,recxw[ntrw:],recyw[ntrw:])
      #plot_dat2d(dat[ntrw:ntrw+nrecw[iexp],:1500],show=False,dt=dt,dmin=dmin,dmax=dmax,pclip=0.01,aspect=50)
      #plot_dat2d(smute[ntrw:ntrw+nrecw[iexp],:1500],dt=dt,dmin=dmin,dmax=dmax,pclip=0.01,aspect=50)
    else:
      smute[ntrw:] = mute_f3shot(dat[ntrw:],srcxw[iexp],srcyw[iexp],nrecw[iexp],strmw[ntrw:],recxw[ntrw:],recyw[ntrw:])
      #plot_dat2d(dat[ntrw:ntrw+nrecw[iexp],:1500],show=False,dt=dt,dmin=dmin,dmax=dmax,pclip=0.01,aspect=50)
      #plot_dat2d(smute[ntrw:ntrw+nrecw[iexp],:1500],dt=dt,dmin=dmin,dmax=dmax,pclip=0.01,aspect=50)
    ntrw += nrecw[iexp]

  # Write out the muted shots
  if(ibtch == 0):
    sep.write_file("f3_shots3interp_full_muted.H",smute.T,ds=[dt,1.0])
  else:
    sep.append_file("f3_shots3interp_full_muted.H",smute.T)

