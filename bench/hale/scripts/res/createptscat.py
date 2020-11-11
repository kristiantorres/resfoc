import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
from scaas.wavelet import ricker
import oway.defaultgeom as geom
from genutils.plot import plot_img2d
from genutils.movie import viewimgframeskey

# IO
sep = seppy.sep()

# Dimensions
nx = 800; dx = 0.015
ny = 1;   dy = 0.125
nz = 400; dz = 0.005

# Constant input slowness
vel = np.zeros([nz,ny,nx],dtype='float32')
vel[:] = 2

# Reflectivity
ref = np.zeros(vel.shape,dtype='float32')
ref[200,0,400] = 1.0
#refsm = smooth(ref,rect1=3,rect3=3)
#plot_img2d(refsm[:,0,:],imin=-1,imax=1,interp='none')
refsm = ref

# Create ricker wavelet
n1 = 2000; d1 = 0.004;
freq = 8; amp = 0.5; dly = 0.2
wav = ricker(n1,d1,freq,amp,dly)

osx = 20; dsx = 10; nsx = 77
wei = geom.defaultgeom(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                       nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

dat = wei.model_data(wav,d1,dly,minf=1.0,maxf=31.0,vel=vel,ref=refsm,time=True,
                     ntx=15,px=112,nthrds=40,sverb=True)

img = wei.image_data(dat,d1,minf=1.0,maxf=31.0,vel=0.9*vel,nhx=20,nthrds=40,sverb=True)
nhx,ohx,dhx = wei.get_off_axis()

imgt = np.ascontiguousarray(np.transpose(img,(0,1,3,4,2))) # [nhy,nhx,nz,ny,nx] -> [nhy,nhx,ny,nx,nz]
sep.write_file("ptscat.H",imgt.T,os=[0,0,ohx],ds=[dz,dx,dhx])

