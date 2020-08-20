import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scipy.ndimage import gaussian_filter
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from resfoc.gain import agc,tpow
from resfoc.resmig import preresmig,get_rho_axis
from genutils.plot import plot_wavelet
from genutils.movie import viewimgframeskey
import matplotlib.pyplot as plt

# Create SEP IO object
sep = seppy.sep()

# Set flags
tconv = False; aconv = False

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
# Model the data
allshot = prp.model_lindata(rvelsm,rref,wav,dtd,verb=True,nthrds=24)

# Image the data
prp.build_taper(100,200)
prp.plot_taper(rref,cmap='gray')

img = prp.wem(rvelsm,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=24)
nh,oh,dh = prp.get_off_axis()

# Residual migration
inro = 10; idro = 0.0025
storm = preresmig(img,[dh,dz,dx],nps=[2049,513,1025],nro=inro,dro=idro,time=tconv,transp=True,verb=True,nthreads=19)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

# Write to file
stormt = np.transpose(storm,(2,3,1,0))
if(tconv): dz = dtd
sep.write_file("resangtrue.H",stormt,os=[0,0,oh,ooro],ds=[dz,dx,dh,odro])

