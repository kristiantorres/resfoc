import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig,convert2time,get_rho_axis
from scaas.off2ang import off2angkzx, get_angkzx_axis

sep = seppy.sep()

# Read in the image
iaxes,img = sep.read_file("wrngposmsked.H")
img = img.reshape(iaxes.n,order='F')
imgt = np.ascontiguousarray(img.T).astype('float32')
imgtw = imgt[:,0,:,:]

# Get axes
[nz,nx,ny,nhx] = iaxes.n; [oz,ox,oy,ohx] = iaxes.o; [dz,dx,dy,dhx] = iaxes.d

# Depth Residual migration
inro = 21; idro = 0.001250
rmig = preresmig(imgtw,[dhx,dx,dz],nps=[65,513,2049],nro=inro,dro=idro,time=False,nthreads=18,verb=True)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

# Convert to time
dtd = 0.004
rmigt = convert2time(rmig,dz,dt=dtd,dro=odro,verb=True)

# Convert to angle
na = 64
rangs  = off2angkzx(rmig ,ohx,dhx,dz,na=na,nthrds=20,transp=True,rverb=True)
rangst = off2angkzx(rmigt,ohx,dhx,dz,na=na,nthrds=20,transp=True,rverb=True)
na,oa,da = get_angkzx_axis(na=na)

sep.write_file("sigsbeewrngposres.H" ,rangs.T,ds=[dz,da,dx,odro],os=[0,oa,ox,ooro])
sep.write_file("sigsbeewrngposrest.H",rangst.T,ds=[dz,da,dx,odro],os=[0,oa,ox,ooro])

