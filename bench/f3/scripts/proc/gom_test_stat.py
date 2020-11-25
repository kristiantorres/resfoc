import inpout.seppy as seppy
import numpy as np
from pef.stat.pef1d import gapped_pef
from pef.stat.conv1dm import conv1dm
from genutils.plot import plot_img2d

sep = seppy.sep()
gaxes,gom = sep.read_file("GoMsp.H")
gom = np.ascontiguousarray(gom.reshape(gaxes.n,order='F').T).astype('float32')
deb = np.zeros(gom.shape,dtype='float32')
ntr,nt = gom.shape

plot_img2d(gom.T,pclip=0.1)

lags,invflt = gapped_pef(gom[500],na=50,gap=20,niter=300)

cop = conv1dm(nt,len(lags),lags,flt=invflt)

for itr in range(ntr):
  cop.forward(False,gom[itr],deb[itr])

plot_img2d(gom.T,pclip=0.1,show=False)
plot_img2d(deb.T,pclip=0.1)

