import inpout.seppy as seppy
import numpy as np
from opt.linopt.cd import cd
from opt.linopt.essops.weight import weight
from scaas.trismooth import smoothop
from resfoc.semb import pick
import matplotlib.pyplot as plt

sep = seppy.sep()

saxes,smb = sep.read_file('mltestdogsmb.H')
smb = np.ascontiguousarray(smb.reshape(saxes.n,order='F').T).astype('float32')

[nz,nro,nx] = saxes.n; [dz,dro,dx] = saxes.d; [oz,oro,ox] = saxes.o

rho = pick(smb,oro,dro,vel0=1.0)

#
#pck2 = np.zeros([nx,nz],dtype='float32')
#ampl = np.zeros([nx,nz],dtype='float32')
#pcko = np.zeros([nx,nz],dtype='float32')
#
#an = 1.0
#gate = 3
#norm = True
#vel0 = 1.0
#
#pickscan(an,gate,norm,vel0,oro,dro,nz,nro,nx,smb,pck2,ampl,pcko)
#
#pckn = oro + pck2*dro
#
taxes,tpk = sep.read_file('tpick.H')
tpk = tpk.reshape(taxes.n,order='F').T
#
##plt.figure(); plt.imshow(tpk.T,cmap='seismic'); plt.colorbar()
##plt.figure(); plt.imshow(pckn.T,cmap='seismic'); plt.colorbar()
##plt.show()
#
#smop = smoothop([nx,nz],rect1=40,rect2=20)
#
#wop = weight(ampl)
#
#sm0 = np.zeros(ampl.shape,dtype='float32')
#
#smf = cd(wop,pcko,sm0,shpop=smop,niter=100)
#
#smf += vel0
#
plt.figure(); plt.imshow(rho.T,cmap='seismic'); plt.colorbar()
plt.figure(); plt.imshow(tpk.T,cmap='seismic'); plt.colorbar()
plt.show()

