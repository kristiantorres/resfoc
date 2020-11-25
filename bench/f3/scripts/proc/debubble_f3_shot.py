import inpout.seppy as seppy
import numpy as np
from pef.stat.pef1d import gapped_pef
from pef.stat.conv1dm import conv1dm
from genutils.ptyprint import progressbar

sep = seppy.sep()
saxes,sht = sep.read_file("f3_shots2_muted.H")
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
deb = np.zeros(sht.shape,dtype='float32')
ntr,nt = sht.shape
dt = 0.002

# Read in geometry
sxaxes,srcx = sep.read_file("/data3/northsea_dutch_f3/f3_srcx2.H")
syaxes,srcy = sep.read_file("/data3/northsea_dutch_f3/f3_srcy2.H")
rxaxes,recx = sep.read_file("/data3/northsea_dutch_f3/f3_recx2.H")
ryaxes,recy = sep.read_file("/data3/northsea_dutch_f3/f3_recy2.H")

naxes,nrec = sep.read_file("/data3/northsea_dutch_f3/f3_nrec2.H")
nrec = nrec.astype('int32')
nsht = len(nrec)

num = 620
tol = 2.0

ntr = 0
for isht in progressbar(range(nsht),"nsht:",verb=True):
  isrcx,isrcy = srcx[isht],srcy[isht]
  # Search for the correct trace to estimate the PEF
  for k in range(ntr,ntr+nrec[isht]):
    dist = np.sqrt((recx[k]-isrcx)**2 + (recy[k]-isrcy)**2)
    if(np.abs(dist - num) < tol):
      # Estimate the PEF and construct the filter
      lags,invflt = gapped_pef(sht[k],na=50,gap=20,niter=300,verb=False)
      cop = conv1dm(nt,len(lags),lags,flt=invflt)
      break
  # Apply to all traces in the shot
  for k in range(ntr,ntr+nrec[isht]):
    cop.forward(False,sht[k],deb[k])
  ntr += nrec[isht]

sep.write_file("f3_shots2_muted_debub_shot.H",deb.T,ds=[dt,1.0])

