import inpout.seppy as seppy
import numpy as np
from genutils.ptyprint import progressbar
from genutils.plot import plot_dat2d

# Read in the headers
sep = seppy.sep()
saxes,srcx = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_srcx3_full.H")
saxes,srcy = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_srcy3_full.H")
naxes,nrec = sep.read_file("/data3/northsea_dutch_f3/windowed_data/f3_nrec3_full.H")
nrec = nrec.astype('int')

# First add the source coordinates to the toremove.txt file
with open('./doc/toremove.txt','r') as f:
  lines = f.readlines()
rmlst = [int(iline.rstrip()) for iline in lines]
rmset = set(rmlst)

with open('./doc/toremovesxsy.txt','w') as f:
  for irm in rmlst:
    sx,sy = srcx[irm],srcy[irm]
    f.write("%d %d %d\n"%(irm,int(sx),int(sy)))

# Read in the data shot by shot
begsht = 0
ctr = np.sum(nrec[:begsht])
for isht in progressbar(range(begsht,len(srcx)),"shots:",verb=True):
  daxes,dat   = sep.read_wind("f3_shots3interp_full.H",fw=ctr,nw=nrec[isht])
  dat = dat.reshape(daxes.n,order='F').astype('float32')
  rxaxes,recx = sep.read_wind("/data3/northsea_dutch_f3/windowed_data/f3_recx3_full.H",fw=ctr,nw=nrec[isht])
  ryaxes,recy = sep.read_wind("/data3/northsea_dutch_f3/windowed_data/f3_recy3_full.H",fw=ctr,nw=nrec[isht])
  staxes,strm = sep.read_wind("/data3/northsea_dutch_f3/windowed_data/f3_strm3_full.H",fw=ctr,nw=nrec[isht])
  if(isht not in rmlst):
    if(isht == begsht):
      sep.write_file("f3_shots3interp_full_clean.H",dat,ds=[0.004,1.0])
      sep.write_file("f3_recx3_full_clean.H",recx)
      sep.write_file("f3_recy3_full_clean.H",recy)
      sep.write_file("f3_strm3_full_clean.H",strm)
      sep.write_file("f3_srcx3_full_clean.H",np.asarray([srcx[isht]],dtype='float32'))
      sep.write_file("f3_srcy3_full_clean.H",np.asarray([srcy[isht]],dtype='float32'))
      sep.write_file("f3_nrec3_full_clean.H",np.asarray([nrec[isht]],dtype='float32'))
    else:
      sep.append_file("f3_shots3interp_full_clean.H",dat)
      sep.append_file("f3_recx3_full_clean.H",recx)
      sep.append_file("f3_recy3_full_clean.H",recy)
      sep.append_file("f3_strm3_full_clean.H",strm)
      sep.append_file("f3_srcx3_full_clean.H",np.asarray([srcx[isht]],dtype='float32'))
      sep.append_file("f3_srcy3_full_clean.H",np.asarray([srcy[isht]],dtype='float32'))
      sep.append_file("f3_nrec3_full_clean.H",np.asarray([nrec[isht]],dtype='float32'))
  ctr += nrec[isht]

