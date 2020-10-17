import inpout.seppy as seppy
import numpy as np
import oway.coordgeom as geom
from scaas.wavelet import ricker
from oway.costaper import costaper
from scaas.trismooth import smooth
import matplotlib.pyplot as plt
from genutils.plot import plot_wavelet

# IO
sep = seppy.sep()

# Read in the velocity and reflectivity models
vaxes,ivel = sep.read_file("hale_veltr3.H")
ivel = np.ascontiguousarray(ivel.reshape(vaxes.n,order='F')).astype('float32')
raxes,iref = sep.read_file("hale_reftr3.H")
iref = np.ascontiguousarray(iref.reshape(raxes.n,order='F')).astype('float32')
[nz,nx] = vaxes.n; [dz,dx] = vaxes.d; [oz,ox] = vaxes.o
ny = 1; dy = 1

# Read in the acquisition geometry
saxes,srcx = sep.read_file("hale_srcxflatbob.H")
raxes,recx = sep.read_file("hale_recxflatbob.H")
_,nrec= sep.read_file("hale_nrecbob.H")
nrec = nrec.astype('int')

# Window
ntr = 48; nsht = 10
nrecw = nrec[:nsht]
srcxw = srcx[:nsht]
recxw = recx[:ntr*nsht]

# Convert velocity to slowness
slo = np.zeros([nz,ny,nx],dtype='float32')
ref = np.zeros([nz,ny,nx],dtype='float32')

# Smooth in slowness
slo[:,0,:] = smooth(1/ivel,rect1=35,rect2=35)
vel = 1/slo

# Build the reflectivity
reftap = costaper(iref,nw1=16)
ref[:,0,:] = reftap

# Create ricker wavelet
n1 = 1500; d1 = 0.004;
freq = 20; amp = 0.5; t0 = 0.2;
wav = ricker(n1,d1,freq,amp,t0)

#plot_wavelet(wav,d1)

# Create the coordgeom object
wei = geom.coordgeom(nx,dx,ny,dy,nz,dz,ox=ox,nrec=nrec,srcxs=srcx,recxs=recx)

# Create wavelet
dat = wei.model_data(wav,d1,t0,minf=1.0,maxf=51.0,vel=vel,nrmax=20,ref=ref,
                     ntx=16,nthrds=40,px=100,eps=0.0)

img = wei.image_data(dat,d1,minf=1.0,maxf=51.0,vel=vel,nrmax=20,nthrds=40,
                     ntx=16)

#ang = wei.to_angle(img,amax=60)

#print(ang.shape)

sep.write_file("hale_trdata3.H",dat.T,ds=[d1,1.0])

sep.write_file("hale_trimg3.H",img,os=[oz,0.0,ox],ds=[dz,dy,dx])

#plt.figure()
#plt.imshow(dat.T,cmap='gray',interpolation='sinc',aspect='auto')
#plt.show()

