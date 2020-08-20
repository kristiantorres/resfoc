import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from resfoc.resmig import preresmig
from scaas.gradtaper import build_taper
from resfoc.resmig import preresmig
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey

sep = seppy.sep()

faxes,fimgo = sep.read_file("./dat/totex/fimgo.H")
fimgo = fimgo.reshape(faxes.n,order='F')
fimgo = np.ascontiguousarray(fimgo.T).astype('float32')

[nz,nx,nh] = faxes.n; [dz,dx,dh] = faxes.d

zo = fimgo[16]

z1t = 70; z2t = 150
tap1d,tap = build_taper(nx,nz,z1t,z2t)

#plt.imshow(agc(zo).T,cmap='gray',interpolation='sinc')
#plt.imshow(tap,cmap='jet',alpha=0.2)
#plt.imshow(agc(tap.T*zo).T,cmap='gray',interpolation='sinc')
#plt.show()

imgn = np.asarray([tap.T*fimgo[ih] for ih in range(nh)])

viewimgframeskey(imgn,pclip=0.1)

rho = 0.9875; dro = 0.00125
rmig  = np.squeeze(preresmig(imgn,[dh,dx,dz],nps=[2049,nx+1,nz+1],nro=1,oro=rho,dro=dro,time=False,nthreads=1,verb=True))

zormig = rmig[16,:,:]

plt.imshow(agc(zormig).T,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
plt.show()

sep.write_file("testtapres.H",rmig.T,ds=faxes.d,os=faxes.o)

