import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
from resfoc.gain import agc
from genutils.ptyprint import progressbar
from genutils.movie import resangframes, viewimgframeskey
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

sep = seppy.sep()
aaxes,ang = sep.read_file("fltimgbigresangwrng.H")
ang = ang.reshape(aaxes.n,order='F')
ang = np.ascontiguousarray(ang.T).astype('float32')

[dz,da,dx,dro] = aaxes.d; [oz,oa,ox,oro] = aaxes.o; [nz,na,nx,nro] = aaxes.n

#t1 = time.time()
## Compute AGC
angagc = np.zeros(ang.shape,dtype='float32')
for iro in progressbar(range(nro),"irho:"):
  angagc[iro] = agc(ang[iro])

tot = time.time() - t1
print("Serial time = %f minutes"%(tot/60.0))

# Split rhos
#csize = 11; beg = 0; end = csize; nchnk = 3
#rhos1 = [ang[iro] for iro in range(beg,end)]
#beg += csize; end += csize
#rhos2 = [ang[iro] for iro in range(beg,end)]
#beg += csize; end += csize
#rhos3 = [ang[iro] for iro in range(beg,end)]
#
#rhos = [rhos1,rhos2,rhos3]
#agcs = []
#
#t1 = time.time()
#run_pool = Pool(csize)
#for ichnk in progressbar(range(nchnk), "ichunk:"):
#  out = run_pool.map(agc,rhos[ichnk])
#  agcs.append(out)
#run_pool.close()
#
#tot = time.time() - t1
#print("Parallel time = %f minutes"%(tot/60.0))

#sep.write_file("fltimgbigresangagc2wrng.H",angagc.T,ds=aaxes.d,os=aaxes.o)

#resangframes(ang,dz/1000.0,dx/1000.0,dro,oro,pclip=0.5,interp='none')

#stack = np.sum(ang,axis=2)

#saxes,stack = sep.read_file("fltimgbigresangstkwrng.H")
#stack = np.ascontiguousarray(stack.reshape(saxes.n,order='F').T).astype('float32')

#stack2 = stack*stack

#semb = smooth(stack2.astype('float32'),rect1=40,rect3=5)

#sembt = np.transpose(semb,(1,2,0))

#print(sembt.shape)

#viewimgframeskey(sembt,transp=False,cmap='jet')

