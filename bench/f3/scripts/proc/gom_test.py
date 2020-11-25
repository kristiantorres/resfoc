import inpout.seppy as seppy
import numpy as np
from pef.nstat.peflms1d import peflmsgap1d
from genutils.plot import plot_img2d

sep = seppy.sep()
gaxes,gom = sep.read_file("GoMsp.H")
gom = np.ascontiguousarray(gom.reshape(gaxes.n,order='F').T).astype('float32')
deb = np.zeros(gom.shape,dtype='float32')
ntr,nt = gom.shape

plot_img2d(gom.T,pclip=0.1)
nw = 50
a = np.zeros([nw],dtype='float32')

for itr in range(ntr):
  err,a = peflmsgap1d(gom[itr],nw=nw,gap=20,mu=0.02,w0=a)
  deb[itr,:] = err[:]

plot_img2d(gom.T,pclip=0.1,show=False)
plot_img2d(deb.T,pclip=0.1)

