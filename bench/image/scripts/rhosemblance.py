import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
from resfoc.gain import agc
from genutils.movie import resangframes, viewimgframeskey
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

sep = seppy.sep()
aaxes,ang = sep.read_file("fltimgbigresang.H")
ang = ang.reshape(aaxes.n,order='F')
ang = np.ascontiguousarray(ang.T).astype('float32')

[dz,da,dx,dro] = aaxes.d; [oz,oa,ox,oro] = aaxes.o; [nz,na,nx,nro] = aaxes.n

angagc = np.asarray(Parallel(n_jobs=24)(delayed(agc)(ang[iro]) for iro in range(nro)))

#sep.write_file("fltimgbigresangagc.H",angagc.T,ds=aaxes.d,os=aaxes.o)

#resangframes(angagc,dz/1000.0,dx/1000.0,dro,oro,pclip=0.5,interp='none')

stack   = np.sum(angagc,axis=2)
stacksq = stack*stack
##stacksq = np.abs(stack*stack)
num = smooth(stacksq.astype('float32'),rect1=10,rect3=2)

sqstack = np.sum(angagc*angagc,axis=2)
denom = smooth(sqstack.astype('float32'),rect1=10,rect3=2)

semb = num/denom
#
##semb = num
##semb = denom
#
##sembt = np.transpose(semb,(1,2,0))
#
##viewimgframeskey(sembt,transp=False,cmap='jet')
#
sembt = np.transpose(semb,(2,0,1))
sep.write_file('rhosemblance.H',sembt,ds=[dz,dro,dx],os=[oz,oro,ox])

