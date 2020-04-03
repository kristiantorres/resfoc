import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scipy.ndimage import gaussian_filter
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from resfoc.gain import agc,tpow
import matplotlib.pyplot as plt
from utils.plot import plot_wavelet
from utils.movie import viewimgframeskey

# Create SEP IO object
sep = seppy.sep()

# Read in the model
vaxes,vel = sep.read_file('../trdat/dat/vels/velflts/velfltmod0000.H')
vel = vel.reshape(vaxes.n,order='F')
velw = np.ascontiguousarray((vel[:,:,0].T).astype('float32'))
# Read in the reflectivity
raxes,ref = sep.read_file('../trdat/dat/vels/velflts/velfltref0000.H')
ref = ref.reshape(raxes.n,order='F')
refw = np.ascontiguousarray((ref[:,:,0].T).astype('float32'))

# Resample the model
nx = 1024; nz = 512
rvel = (resample(velw,[nx,nz],kind='linear')).T
rref = (resample(refw,[nx,nz],kind='linear')).T
dz = 10; dx = 10

# Create migration velocity
rvelsm = gaussian_filter(rvel,sigma=20)

dsx = 20; bx = 25; bz = 25
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=52,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(rvelsm,cmap='jet',show=False)
prp.plot_acq(rref,cmap='gray',show=False)

# Create data axes
ntu = 6400; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)
plot_wavelet(wav,dtu)

dtd = 0.004
allshot = prp.model_lindata(rvelsm,rref,wav,dtd,verb=True,nthrds=24)

viewimgframeskey(allshot,transp=False,pclip=0.2)

## Write out all shots
datout = np.transpose(allshot,(1,2,0))
sep.write_file('fltdata.H',datout,ds=[dtd,dx,dsx])

sep.to_header("fltdata.H","srcz=%d recz=%d"%(0,0))
sep.to_header("fltdata.H","bx=%d bz=%d alpha=%f"%(bx,bz,0.99))

