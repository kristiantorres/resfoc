import inpout.seppy as seppy
import numpy as np
from resfoc.pickscan import pick as pick
import matplotlib.pyplot as plt

sep = seppy.sep()

saxes,smb = sep.read_file('mltestdogsmb.H')
smb = np.ascontiguousarray(smb.reshape(saxes.n,order='F').T).astype('float32')

[nz,nro,nx] = saxes.n; [dz,dro,dx] = saxes.d; [oz,oro,ox] = saxes.o

pck2 = np.zeros([nx,nz],dtype='float32')
ampl = np.zeros([nx,nz],dtype='float32')
pcko = np.zeros([nx,nz],dtype='float32')

an = 1.0
gate = 3
norm = True
vel0 = 1.0

pick(an,gate,norm,vel0,oro,dro,nz,nro,nx,smb,pck2,ampl,pcko)

pckn = oro + pcko*dro

taxes,tpk = sep.read_file('tpick.H')
tpk = tpk.reshape(taxes.n,order='F').T

plt.figure(); plt.imshow(tpk.T,cmap='seismic')
plt.figure(); plt.imshow(pckn.T,cmap='seismic')
plt.show()
