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
vaxes,vel = sep.read_file('velrug.H')
vel = vel.reshape(vaxes.n,order='F')
velw = np.ascontiguousarray(vel.astype('float32'))
# Read in the reflectivity
raxes,ref = sep.read_file('lyrrug.H')
ref = ref.reshape(raxes.n,order='F')
refw = np.ascontiguousarray(ref.astype('float32'))
# Read in the data
daxes,dat = sep.read_file('rugsmall.H')
dat = dat.reshape(daxes.n,order='F')
allshot = np.ascontiguousarray(np.transpose(dat,(2,0,1)).astype('float32'))
dtd = daxes.d[0]

nz = vaxes.n[0]; dz = vaxes.d[0]
nx = vaxes.n[1]; dx = vaxes.d[1]

# Create migration velocity
rvelsm = smooth(velw,rect1=20,rect2=20)

dsx = 5; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=52,dsx=dsx,bx=bx,bz=bz)

#prp.plot_acq(rvelsm,cmap='jet',show=False)
#prp.plot_acq(refw,cmap='gray',show=False,vmin=-1,vmax=1)

# Create data axes
ntu = 3000; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)
#plot_wavelet(wav,dtu)

prp.build_taper(100,200)
prp.plot_taper(refw,cmap='gray',vmin=-1,vmax=1)

img = prp.wem(rvelsm,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=24)
nh,oh,dh = prp.get_off_axis()

#gimg = tpow(img,dz,tpow=2)

#viewimgframeskey(gimg,transp=False)

# Write out image
imgo = np.transpose(img,(1,2,0))
sep.write_file('rugimgsmall.H',imgo,ors=[0.0,0.0,oh],ds=[dz,dx,dh])

