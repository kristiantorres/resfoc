import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import rand_preresmig, convert2time, get_rho_axis
from deeplearn.utils import next_power_of_2
from scaas.off2ang import off2angkzx, get_angkzx_axis

sep = seppy.sep()

# Read in the extended image
iaxes,img = sep.read_file("sigextimg.H")
[nz,nx,ny,nhx] = iaxes.n; [dz,dx,dy,dhx] = iaxes.d; [oz,ox,oy,ohx] = iaxes.o
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')
imgw = img[:,0,:,:]

# Residual migration axis
nro = 21; dro = 0.001250

# Do the random residual migration
nsin  = [nhx,nx,nz]
nps = [next_power_of_2(nin)+1 for nin in nsin]
rmig,rho = rand_preresmig(imgw,[dhx,dx,dz],nps=nps,nro=nro,dro=dro,verb=False)
rmigt = convert2time(rmig,dz,dt=0.008,oro=rho,dro=dro,verb=False)[0]
rmige = rmigt[np.newaxis,:,np.newaxis,:,:]

# Convert to angle
na=64
rang = off2angkzx(rmige,ohx,dhx,dz,na=na,nthrds=10,transp=True,cverb=False)
na,oa,da = get_angkzx_axis(na,amax=60)
rangc = rang[0]

sep.write_file("sigsbee_randresmig.H",rmige.T,os=iaxes.o,ds=iaxes.d)
sep.write_file("sigsbee_randrho1.H",np.asarray([rho]))
sep.write_file("sigsbee_randrang.H",rangc.T,os=[oz,oa,0.0,ox],ds=[dz,da,1.0,dx])

