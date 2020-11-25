import inpout.seppy as seppy
import numpy as np
from pef.stat.pef1d import pef1d, pef1dmask
from pef.stat.conv1dm import conv1dm
from opt.linopt.combops import chainop
from opt.linopt.cd import cd
from opt.linopt.essops.identity import identity
import matplotlib.pyplot as plt

sep = seppy.sep()
baxes,bub = sep.read_file("nucleus.H")
bub = bub.astype('float32')
nt = len(bub)
deb = np.zeros(bub.shape,dtype='float32')

lags = np.arange(69,300,1).astype('int32')
lags[0] = 0

nlag= len(lags)
pef = pef1d(nt,nlag,lags,aux=bub)
idat = pef.create_data()

flt = np.zeros(nlag,dtype='float32')
flt[0] = 1.0

mask = pef1dmask()
idop = identity()
zro = np.zeros(nlag,dtype='float32')
dkop = chainop([mask,pef],pef.get_dims())

pefres = []
invflt = cd(dkop,idat,flt,niter=300,rdat=zro,eps=0.01,ress=pefres)

cop = conv1dm(nt,nlag,lags,flt=invflt)

cop.forward(False,bub,deb)

plt.plot(pefres[-1])
plt.plot(deb); plt.show()

#plt.plot(invflt); plt.show()

