import inpout.seppy as seppy
import numpy as np
from pef.stat.pef1d import gapped_pef
from pef.stat.conv1dm import conv1dm
from genutils.plot import plot_dat2d
from genutils.ptyprint import progressbar

sep = seppy.sep()
saxes,sht = sep.read_file("f3_shots3interp_full_muted.H")
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
deb = np.zeros(sht.shape,dtype='float32')
ntr,nt = sht.shape
dt = 0.004

lags,invflt = gapped_pef(sht[20],na=30,gap=10,niter=300,verb=False)
cop = conv1dm(nt,len(lags),lags,flt=invflt)

for itr in progressbar(range(ntr),"ntr:",verb=True):
  cop.forward(False,sht[itr],deb[itr])
  #if(itr == 960): 
  #  plot_dat2d(sht[:itr,:750],dt=dt,pclip=0.05,aspect=50,show=False)
  #  plot_dat2d(deb[:itr,:750],dt=dt,pclip=0.05,aspect=50,show=True)

del sht
sep.write_file("f3_shots3interp_full_muted_debub_onetr.H",deb.T,ds=[dt,1.0])

