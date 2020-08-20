import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scaas.trismooth import smooth
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from resfoc.gain import agc,tpow
from resfoc.resmig import preresmig,get_rho_axis
import matplotlib.pyplot as plt
from genutils.plot import plot_wavelet
from genutils.movie import viewimgframeskey

# Create SEP IO object
sep = seppy.sep()

# Read in the model
vaxes,vel = sep.read_file('../fltdat/velbig.H')
vel = vel.reshape(vaxes.n,order='F')
velw = np.ascontiguousarray(vel.astype('float32'))
# Read in the reflectivity
raxes,ref = sep.read_file('../fltdat/refbig.H')
ref = ref.reshape(raxes.n,order='F')
refw = np.ascontiguousarray(ref.astype('float32'))

nz = vaxes.n[0]; dz = vaxes.d[0]
nx = vaxes.n[1]; dx = vaxes.d[1]

# Create migration velocity
rvelsm = smooth(velw,rect1=30,rect2=30)

dsx = 20; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=51,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(rvelsm,cmap='jet',show=False)
prp.plot_acq(refw,cmap='gray',show=False)

# Create data axes
ntu = 6500; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)
plot_wavelet(wav,dtu)

dtd = 0.004
allshot = prp.model_lindata(rvelsm,refw,wav,dtd,verb=True,nthrds=24)

viewimgframeskey(allshot,transp=False,pclip=0.2)

## Write out all shots
datout = np.transpose(allshot,(1,2,0))
sep.write_file('fltbig.H',datout,ds=[dtd,dx,dsx])

sep.to_header("fltbig.H","srcz=%d recz=%d"%(0,0))
sep.to_header("fltbig.H","bx=%d bz=%d alpha=%f"%(bx,bz,0.99))

