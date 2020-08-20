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
nxr = int(nx/1)
velr,[dxr,dz]= resample(vel, [nz,nxr], ds=[dz,dx],kind='cubic')
refr = resample(ref, [nz,nxr], kind='cubic').astype('float32')
ptbr = resample(ptb, [nz,nxr], kind='cubic').astype('float32')

# Convert velocity to slowness
slo = np.zeros([nz,ny,nxr],dtype='float32')

# Smooth in slowness
slo[:,0,:] = smooth(1/velr,rect1=30,rect2=30)
velmig = 1/slo

veltru = np.zeros(velmig.shape,dtype='float32')

veltru[:,0,:] = velmig[:,0,:] - ptbr

# Build the reflectivity
reftap = costaper(refr,nw1=10)

# Create ricker wavelet
n1 = 2000; d1 = 0.004;
freq = 20; amp = 0.5; t0 = 0.2;
wav = ricker(n1,d1,freq,amp,t0)

plot_wavelet(wav,d1,hspace=0.35)

# Acquisition geometry
nsx = 64; dsx = 20; nsy = 1; dsy = 1; osx = dsx
wei = geom.defaultgeom(nx=nxr,dx=dxr,ny=ny,dy=dy,nz=nz,dz=dz,
                       nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

wei.plot_acq(veltru)

#wei.test_freq_axis(n1,d1,minf=1,maxf=51)

print("Modeling:")
dat = wei.model_data(wav,d1,t0,minf=1.0,maxf=51.0,vel=veltru,nrmax=10,ref=reftap,
                     ntx=15,nthrds=40,px=100,eps=0.0)

print("Imaging:")
imgw  = wei.image_data(dat,d1,minf=1.0,maxf=51.0,vel=velmig,nhx=20,nrmax=6,ntx=15,nthrds=40,eps=0.0)
imgwt = np.transpose(imgw,(2,4,3,1,0)) # [nhy,nhx,nz,ny,nx] -> [nz,nx,ny,nhx,nhy]
#imgwa = wei.to_angle(imgw,nthrds=26,verb=True) # [nhy,nhx,nz,ny,nx] -> [nx,na,nz]

print("Imaging:")
imgr  = wei.image_data(dat,d1,minf=1.0,maxf=51.0,vel=veltru,nhx=20,nrmax=10,ntx=15,nthrds=40,eps=0.0)
imgrt = np.transpose(imgr,(2,4,3,1,0)) # [nhy,nhx,nz,ny,nx] -> [nz,nx,ny,nhx,nhy]
#imgra = wei.to_angle(imgr,nthrds=26,verb=True)

#plt.figure()
#plt.imshow(img[:,0,:],cmap='gray',extent=[0,nxr*dxr,nz*dz,0],interpolation='sinc')
#plt.show()

nhx,ohx,dhx = wei.get_off_axis()
#na,oa,da = wei.get_ang_axis()

sep.write_file("imgowayextrshrteps0.H",imgrt,os=[0,0,0,ohx,0],ds=[dz,dy,dxr,dhx,1.0])
sep.write_file("imgowayextwshrteps0.H",imgwt,os=[0,0,0,ohx,0],ds=[dz,dy,dxr,dhx,1.0])

#sep.write_file("imgowayangrshrt.H",imgra.T,os=[0,oa,0],ds=[dz,da,dx])
#sep.write_file("imgowayangwshrt.H",imgwa.T,os=[0,oa,0],ds=[dz,da,dx])

