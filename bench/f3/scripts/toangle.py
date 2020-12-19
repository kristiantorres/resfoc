import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt
from scaas.off2ang import off2angkzx, get_angkzx_axis

sep = seppy.sep()

caxes,cub = sep.read_file("f3extimgs/f3extimgio5m_tot.H")
nz,nx,ny,nhx = caxes.n; dz,dx,dy,dhx = caxes.d; oz,ox,oy,ohx = caxes.o
cub = cub.reshape(caxes.n,order='F').T
cubn = cub[np.newaxis]

#img = np.zeros([1,nhx,1,nx,nz],dtype='float32')
#img[0,:,0,:,:] = cubn[0,:,40,:,:]

na = 41
ang = off2angkzx(cubn,ohx,dhx,dz,na=na,transp=True,cverb=True)
na,oa,da = get_angkzx_axis(na=na)

sep.write_file("f3ang_tot.H",ang.T,os=[0,oa,0.0,ox,oy],ds=[dz,da,1.0,dx,dy])

