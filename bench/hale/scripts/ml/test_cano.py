import inpout.seppy as seppy
import numpy as np
from scaas.velocity import create_randomptbs_loc
from scaas.trismooth import smooth
import matplotlib.pyplot as plt

sep = seppy.sep()

vaxes,vels = sep.read_file("hale_trvels.H")
vels = np.ascontiguousarray(vels.reshape(vaxes.n,order='F').T).astype('float32')
[nz,nx,nm] = vaxes.n

maxcx = int(0.75*nx); mincx = int(0.25*nx)
maxcz = int(0.22*nz); mincz = int(0.13*nz)

for ivel in range(nm):
  sm = smooth(vels[ivel],rect1=35,rect2=35)
  ano = create_randomptbs_loc(nz,nx,nptbs=3,romin=0.95,romax=1.05,
                              minnaz=100,maxnaz=150,minnax=100,maxnax=400,mincz=mincz,maxcz=maxcz,mincx=mincx,maxcx=maxcx,
                              mindist=100,nptsz=2,nptsx=2,octaves=2,period=80,persist=0.2,ncpu=1,sigma=20)
  ptb = sm.T*ano
  plt.figure()
  plt.imshow(ptb,cmap='jet',interpolation='bilinear',aspect='auto')
  plt.figure()
  plt.imshow(sm.T,cmap='jet',interpolation='bilinear',aspect='auto')
  plt.show()

