import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
import matplotlib.pyplot as plt

sep = seppy.sep()

laxes,lng= sep.read_file("./dat/refocus/mltest/mltestdogangmask.H")
[nz,na,nx,nro] = laxes.n; [dz,da,dx,dro] = laxes.d; [oz,da,ox,oro] = laxes.o
lng = lng.reshape(laxes.n,order='F').T

lng = lng.reshape([nx*nro,na,nz])

angint = resample(lng,[64,nz],kind='quintic')

#plt.imshow(angint[0].T,cmap='gray',interpolation='sinc')
#plt.show()

angintr = angint.reshape([nro,nx,64,nz])

sep.write_file("mltestdogangmaskint.H",angintr.T,os=laxes.o,ds=laxes.d)

