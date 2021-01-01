import inpout.seppy as seppy
import numpy as np
from genutils.ptyprint import progressbar
from genutils.plot import plot_dat2d

# Read in the headers
sep = seppy.sep()
saxes,srcx = sep.read_file("f3_srcx3_full_clean.H")
saxes,srcy = sep.read_file("f3_srcy3_full_clean.H")
naxes,nrec = sep.read_file("f3_nrec3_full_clean.H")
nrec = nrec.astype('int')

# Read in the data shot by shot
begsht = 0
srcs = set()
ctr = np.sum(nrec[:begsht])
for isht in progressbar(range(begsht,len(srcx)),"shots:",verb=True):
  daxes,dat   = sep.read_wind("f3_shots3interp_full_clean.H",fw=ctr,nw=nrec[isht])
  dat = dat.reshape(daxes.n,order='F').astype('float32')
  rxaxes,recx = sep.read_wind("f3_recx3_full_clean.H",fw=ctr,nw=nrec[isht])
  ryaxes,recy = sep.read_wind("f3_recy3_full_clean.H",fw=ctr,nw=nrec[isht])
  staxes,strm = sep.read_wind("f3_strm3_full_clean.H",fw=ctr,nw=nrec[isht])
  if((srcy[isht],srcx[isht]) not in srcs):
    if(isht == begsht):
      sep.write_file("f3_shots3interp_full_clean2.H",dat,ds=[0.004,1.0])
      sep.write_file("f3_recx3_full_clean2.H",recx)
      sep.write_file("f3_recy3_full_clean2.H",recy)
      sep.write_file("f3_strm3_full_clean2.H",strm)
      sep.write_file("f3_srcx3_full_clean2.H",np.asarray([srcx[isht]],dtype='float32'))
      sep.write_file("f3_srcy3_full_clean2.H",np.asarray([srcy[isht]],dtype='float32'))
      sep.write_file("f3_nrec3_full_clean2.H",np.asarray([nrec[isht]],dtype='float32'))
      srcs.add((srcy[isht],srcx[isht]))
    else:
      sep.append_file("f3_shots3interp_full_clean2.H",dat)
      sep.append_file("f3_recx3_full_clean2.H",recx)
      sep.append_file("f3_recy3_full_clean2.H",recy)
      sep.append_file("f3_strm3_full_clean2.H",strm)
      sep.append_file("f3_srcx3_full_clean2.H",np.asarray([srcx[isht]],dtype='float32'))
      sep.append_file("f3_srcy3_full_clean2.H",np.asarray([srcy[isht]],dtype='float32'))
      sep.append_file("f3_nrec3_full_clean2.H",np.asarray([nrec[isht]],dtype='float32'))
      srcs.add((srcy[isht],srcx[isht]))
  else:
    print("%d %d is a duplicate"%(srcy[isht],srcx[isht]))
  ctr += nrec[isht]

