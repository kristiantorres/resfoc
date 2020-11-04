import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig,convert2time,get_rho_axis
from scaas.gradtaper import build_taper, build_taper_ds, build_taper_bot
from scaas.off2ang import off2angkzx, get_angkzx_axis
from genutils.plot import plot_img2d

sep = seppy.sep()

# Read in the defocused image
iaxes,img = sep.read_file("spimgextbobdistrwrng.H")
img  = img.reshape(iaxes.n,order='F')
imgt = np.ascontiguousarray(img.T).astype('float32')
imgtw = imgt[:,0,30:542,:450]

# Get axes
nhx,nx,nz = imgtw.shape; oz,ox,oy,ohx = iaxes.o; dz,dx,dy,dhx = iaxes.d

# Taper the top of the image
_,tap = build_taper_bot(nx,nz,400,430)

timgtw = np.zeros(imgtw.shape,dtype='float32')
for ihx in range(nhx):
  timgtw[ihx] = tap.T*imgtw[ihx]

sc = 0.5
imin = sc*np.min(timgtw); imax = sc*np.max(timgtw)

plot_img2d(timgtw[20].T,aspect='auto')

# Depth Residual migration
inro = 81; idro = 0.001250
rmig = preresmig(timgtw,[dhx,dx,dz],nps=[65,512,512],nro=inro,dro=idro,time=False,nthreads=18,verb=True)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

# Convert to time
dtd = 0.004
rmigt = convert2time(rmig,dz,dt=dtd,dro=odro,verb=True)

na = 64
rangs  = off2angkzx(rmig ,ohx,dhx,dz,na=na,nthrds=20,transp=True,rverb=True)
na,oa,da = get_angkzx_axis(na=na)
sep.write_file("resfaultfocuswind.H",rangs.T,ds=[dz,da,dx,odro],os=[oz,oa,7.37,ooro])
del rangs; del rmig

rangst = off2angkzx(rmigt,ohx,dhx,dz,na=na,nthrds=20,transp=True,rverb=True)
sep.write_file("resfaultfocuswindt.H",rangst.T,ds=[dz,da,dx,odro],os=[oz,oa,7.37,ooro])

