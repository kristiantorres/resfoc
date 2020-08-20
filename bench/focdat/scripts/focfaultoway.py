import inpout.seppy as seppy
import numpy as np
import oway.defaultgeom as geom
from oway.costaper import costaper
from scaas.wavelet import ricker
from scaas.trismooth import smooth
import matplotlib.pyplot as plt
from genutils.plot import plot_wavelet

# File IO
sep = seppy.sep()

# Read in model, reflectivity and perturbation
mltest = np.load("./dat/mltest.npy",allow_pickle=True)[()]
vel = mltest[0]*0.001; ref = mltest[1]; ptb = mltest[2]*0.001

# Spatial axes
nz = vel.shape[0]; nx = vel.shape[1]; ny = 1; nhx = 16; nhy = 1
dz = 0.01; dx = 0.01; dy = 1.0; dhx = 0.01

# Convert velocity to slowness
slo = np.zeros([nz,ny,nx],dtype='float32')

# Smooth in slowness
slo[:,0,:] = smooth(1/vel,rect1=30,rect2=30)
velmig = 1/slo

veltru = np.zeros(velmig.shape,dtype='float32')

veltru[:,0,:] = velmig[:,0,:] - ptb

# Build the reflectivity
reftap = costaper(ref,nw1=20)

# Create ricker wavelet
n1 = 2000; d1 = 0.004;
freq = 20; amp = 0.5; t0 = 0.2;
wav = ricker(n1,d1,freq,amp,t0)

#plot_wavelet(wav,d1,hspace=0.35)

# Acquisition geometry
nsx = 64; dsx = 20; nsy = 1; dsy = 1; osx = dsx
wei = geom.defaultgeom(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                       nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

wei.plot_acq(veltru)

#wei.test_freq_axis(n1,d1,minf=1,maxf=51)


dat = wei.model_data(wav,d1,t0,minf=1.0,maxf=51.0,vel=veltru,nrmax=10,ref=reftap,
                     ntx=15,nthrds=40,wverb=False,px=50)

#daxes,dat = sep.read_file("fltdatoway.H")
#dat = dat.reshape(daxes.n,order='F').T
#datin = np.ascontiguousarray(dat.reshape([1,nsx,ny,nx,n1]))

img = wei.image_data(dat,d1,minf=1.0,maxf=51.0,vel=veltru,nrmax=10,ntx=15,nthrds=40)

sep.write_file("imgdefocoway2w.H",img,os=[0,0,0],ds=[dz,dy,dx])

#sep.write_file("fltdatoway.H",dat.T,os=[0,0,0,osx,0],ds=[d1,dx,dy,dsx,1.0])


