import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scipy.ndimage import gaussian_filter
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from resfoc.gain import tpow
import matplotlib.pyplot as plt
from utils.plot import plot_wavelet
from utils.movie import viewimgframeskey

# Set up IO
sep = seppy.sep()

# Read in the model
vaxes,vel = sep.read_file('../trdat/dat/vels/velflts/velfltmod0000.H')
vel = vel.reshape(vaxes.n,order='F')
velw = np.ascontiguousarray((vel[:,:,0].T).astype('float32'))
# Read in the reflectivity
raxes,ref = sep.read_file('../trdat/dat/vels/velflts/velfltref0000.H')
ref = ref.reshape(raxes.n,order='F')
refw = np.ascontiguousarray((ref[:,:,0].T).astype('float32'))
# Read in the data
daxes,dat = sep.read_file('fltdat.H')
dat = dat.reshape(daxes.n,order='F')
allshot = np.ascontiguousarray(np.transpose(dat,(2,0,1)).astype('float32'))
dtd = daxes.d[0]
# Oneshot for testing
oneshot = allshot[0:2]

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

# Image the data
prp.build_taper(100,200)
prp.plot_taper(rref,cmap='gray')

img = prp.wem(rvelsm,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=24)
nh,oh,dh = prp.get_off_axis()

viewimgframeskey(img,pclip=0.2,transp=False)

# Write out image
imgo = np.transpose(img,(1,2,0))
iaxes = seppy.axes([nz,nx,nh],[0.0,0.0,oh],[dz,dx,dh])
sep.write_file('fltimgextprcnew2.H',imgo,ofaxes=iaxes)

