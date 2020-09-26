import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig,convert2time,get_rho_axis
from scaas.off2ang import off2angkzx, get_angkzx_axis

sep = seppy.sep()

iaxes,img = sep.read_file("intro_img.H")
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')
imgw = img[:,0,:,:]

# Get axis
[nz,nx,ny,nhx] = iaxes.n; [oz,ox,oy,ohx] = iaxes.o; [dz,dx,dy,dhx] = iaxes.d

# Depth Residual migration
inro = 21; idro = 0.001250
rmig = preresmig(imgw,[dhx,dx,dz],nps=[65,1025,513],nro=inro,dro=idro,time=False,nthreads=18,verb=True)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

# Convert to time
dtd = 0.004
rmigt = convert2time(rmig,dz,dt=dtd,dro=odro,verb=True)

# Convert to angle
na = 64
rangs  = off2angkzx(rmig ,ohx,dhx,dz,na=na,nthrds=20,transp=True,rverb=True)
rangst = off2angkzx(rmigt,ohx,dhx,dz,na=na,nthrds=20,transp=True,rverb=True)
na,oa,da = get_angkzx_axis(na=na)

sep.write_file("intro_resang.H" ,rangs.T,ds=[dz,da,dx,odro],os=[0,oa,ox,ooro])
sep.write_file("intro_resangt.H",rangst.T,ds=[dz,da,dx,odro],os=[0,oa,ox,ooro])
