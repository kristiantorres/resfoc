import inpout.seppy as seppy
import numpy as np
from scaas.velocity import insert_circle
from scaas.wavelet import ricker
from scaas.trismooth import smooth
import oway.defaultgeom as geom
import matplotlib.pyplot as plt

sep = seppy.sep()

nz,nx,ny = 512,1024,1
dz,dx,dy = 0.01,0.01,0.01
vel = np.zeros([nz,ny,nx],dtype='float32') + 3

velcirc = insert_circle(vel[:,0,:],dz,dx,centerx=5.12,centerz=2.56,rad=0.025,val=2.5)
plt.imshow(velcirc,cmap='jet'); plt.show()

# Compute reflectivity
ref   = np.zeros([nz,nx],dtype='float32')
for iz in range(nz-1):
  for ix in range(nx):
    ref[iz,ix] = velcirc[iz+1,ix] - velcirc[iz,ix]

refsm = np.zeros([nz,ny,nx],dtype='float32')
refsm[:,0,:] = ref

#ref[254,510] = 1.0
#ref[255,510] = 1.0
#ref[254,511] = 1.0
#ref[255,511] = 1.0
#refsm[:,0,:] = smooth(ref,rect1=3,rect2=3)

plt.imshow(refsm[:,0,:],cmap='gray'); plt.show()

# Acquisition geometry
nsx = 50; dsx = 20; nsy = 1; dsy = 1; osx = dsx
wei = geom.defaultgeom(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                       nsx=nsx,dsx=dsx,osx=dsx,nsy=1,dsy=1.0)

wei.plot_acq(vel)

# Create ricker wavelet
n1 = 2000; d1 = 0.004;
freq = 20; amp = 0.5; t0 = 0.2;
wav = ricker(n1,d1,freq,amp,t0)

print("Modeling")
dat = wei.model_data(wav,d1,t0,minf=1.0,maxf=51.0,vel=vel,nrmax=5,ref=ref,
                     ntx=15,nthrds=40,px=100,eps=0.0)
#plt.imshow(dat[0,25,0].T,cmap='gray',interpolation='sinc'); plt.show()
#print(dat.shape)

print("Imaging")
img  = wei.image_data(dat,d1,minf=1.0,maxf=51.0,vel=vel,nhx=20,nrmax=5,ntx=15,nthrds=40,eps=0.0)
imgt = np.transpose(img,(2,4,3,1,0))

nhx,ohx,dhx = wei.get_off_axis()
sep.write_file("intro_img.H",imgt,os=[0,0,0,ohx,0],ds=[dz,dy,dx,dhx,1.0])


