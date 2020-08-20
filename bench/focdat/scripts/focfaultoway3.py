import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
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

# Lateral resampling of reflectivity and velocity
nxr = int(nx/2)
velr,[dxr,dz]= resample(vel, [nz,nxr], ds=[dz,dx],kind='cubic')
refr = resample(ref, [nz,nxr], kind='cubic').astype('float32')
ptbr = resample(ptb, [nz,nxr], kind='cubic').astype('float32')

# Convert velocity to slowness
slo = np.zeros([nz,ny,nxr],dtype='float32')

# Smooth in slowness
slo[:,0,:] = smooth(1/velr,rect1=30,rect2=30)
velmig = 1/slo


# Build the reflectivity
reftap = costaper(refr,nw1=10)

# Create ricker wavelet
n1 = 2000; d1 = 0.004;
freq = 20; amp = 0.5; t0 = 0.2;
wav = ricker(n1,d1,freq,amp,t0)

#plot_wavelet(wav,d1,hspace=0.35)

# Acquisition geometry
nsx = 32; dsx = 20; nsy = 1; dsy = 1; osx = dsx
wei = geom.defaultgeom(nx=nxr,dx=dxr,ny=ny,dy=dy,nz=nz,dz=dz,
                       nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

wei.plot_acq(velmig)

#wei.test_freq_axis(n1,d1,minf=1,maxf=51)

print("Modeling:")
dat = wei.model_data(wav,d1,t0,minf=1.0,maxf=51.0,vel=velmig,nrmax=10,ref=reftap,
                     ntx=15,nthrds=40,px=100)

print("Imaging:")
img = wei.image_data(dat,d1,minf=1.0,maxf=51.0,vel=velmig,nhx=20,nrmax=6,ntx=15,nthrds=40)

nhx,ohx,dhx = wei.get_off_axis()

imgt = np.transpose(img,(2,4,3,1,0))
sep.write_file("imgowayext.H",imgt,os=[0,0,0,ohx,0],ds=[dz,dy,dxr,dhx,1.0])

