import inpout.seppy as seppy
import numpy as np
from resfoc.semb import rho_semb,pick
import matplotlib.pyplot as plt

sep = seppy.sep()

#aaxes,ang = sep.read_file("./dat/refocus/mltest/mltestdogang.H")
#ang = np.ascontiguousarray(ang.reshape(aaxes.n,order='F').T).astype('float32') # [nro,nx,na,nz]
#nz,na,nx,nro = aaxes.n; dz,da,dx,dro = aaxes.d; oz,oa,ox,oro = aaxes.o

#semb = rho_semb(ang,nthreads=24)

saxes,smb = sep.read_file("./sembtest.H")
smb = np.ascontiguousarray(smb.reshape(saxes.n,order='F').T).astype('float32')
[nz,nro,nx] = saxes.n; [oz,oro,ox] = saxes.o; [dz,dro,dx] = saxes.d

rho = pick(smb,oro,dro,vel0=1.0,norm=False,verb=True)
rhon = pick(smb,oro,dro,vel0=1.0,norm=True,verb=True)

plt.figure(); plt.imshow(rho.T,cmap='seismic'); plt.colorbar();
plt.figure(); plt.imshow(rhon.T,cmap='seismic'); plt.colorbar(); plt.show()

#sep.write_file("sembtest.H",semb.T,os=[oz,oro,ox],ds=[dz,dro,dx])
sep.write_file("rhotest.H",rho.T,os=[oz,ox],ds=[dz,dx])

