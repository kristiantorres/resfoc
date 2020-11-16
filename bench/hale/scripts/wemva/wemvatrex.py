import inpout.seppy as seppy
import numpy as np
import oway.coordgeom as geom
from scaas.trismooth import smooth
from oway.costaper import costaper
from scaas.wavelet import ricker
from genutils.plot import plot_img2d, plot_vel2d

sep = seppy.sep()
# Velocity
vaxes,ivel = sep.read_wind("hale_trvels.H",nw=1,fw=0)
nz,nx = vaxes.n; dz,dx,dm = vaxes.d; oz,ox,om = vaxes.o
ny = 1; oy = 0.0; dy = 1
ivel = np.ascontiguousarray(ivel.reshape(vaxes.n,order='F').T).astype('float32')

# Reflectivity
raxes,iref = sep.read_wind("hale_trrefs.H",nw=1,fw=0)
iref = np.ascontiguousarray(iref.reshape(raxes.n,order='F').T).astype('float32')

# Anomaly
aaxes,iano = sep.read_wind("hale_tranos.H",nw=1,fw=0)
iano = np.ascontiguousarray(iano.reshape(aaxes.n,order='F').T).astype('float32')

# Read in acqusition geometry
saxes,srcx = sep.read_file("hale_srcxflatbob.H")
raxes,recx = sep.read_file("hale_recxflatbob.H")
_,nrec= sep.read_file("hale_nrecbob.H")
nrec = nrec.astype('int')

# Window
ntr = 48; nsht = 50
nrecw = nrec[:nsht]
srcxw = srcx[:nsht]
recxw = recx[:ntr*nsht]

# Convert velocity to slowness
slo = np.zeros([nz,ny,nx],dtype='float32')
ano = np.zeros([nz,ny,nx],dtype='float32')
ref = np.zeros([nz,ny,nx],dtype='float32')

# Smooth in slowness
slo[:,0,:] = smooth(1/ivel,rect1=40,rect2=30).T
vel = 1/slo

# Build the reflectivity
reftap = costaper(iref,nw2=15)
ref[:,0,:] = reftap.T

# Slowness perturbation
ano[:,0,:] = iano.T
dslo = (vel - vel*ano)*4e-04

# Create ricker wavelet
n1 = 1500; d1 = 0.004;
freq = 20; amp = 0.5; t0 = 0.2;
wav = ricker(n1,d1,freq,amp,t0)

#plot_vel2d(vel[:,0,:])
#plot_img2d(ref[:,0,:])

# Build the wave-equation object
wei = geom.coordgeom(nx,dx,ny,dy,nz,dz,ox=ox,nrec=nrec,srcxs=srcx,recxs=recx)

# Create wavelet
dat = wei.model_data(wav,d1,t0,minf=1.0,maxf=51.0,vel=vel,nrmax=20,ref=ref,
                     ntx=16,nthrds=40,px=100)

#print(dat.shape)
#plot_img2d(dat.T,aspect='auto')

img = wei.image_data(dat,d1,minf=1.0,maxf=51.0,vel=vel,nrmax=20,nthrds=40,ntx=16)
sep.write_file("wemvaimg.H",img,os=[oz,oy,ox],ds=[dz,dy,dx])

# Forward WEMVA applied to slowness perturbation perturbation
dimg = wei.fwemva(dslo,dat,d1,minf=1.0,maxf=51.0,vel=vel,nrmax=20,nthrds=40,ntx=16)
sep.write_file("wemvadimg.H",dimg,os=[oz,oy,ox],ds=[dz,dy,dx])
dbck = wei.awemva(dimg,dat,d1,minf=1.0,maxf=51.0,vel=vel,nrmax=20,nthrds=40,ntx=16)
sep.write_file("wemvadslo.H",dbck,os=[oz,oy,ox],ds=[dz,dy,dx])

#plot_img2d(img[:,0,:],aspect='auto',show=False)
#plot_img2d(dimg[:,0,:],aspect='auto',show=False)
#plot_img2d(dbck[:,0,:],aspect='auto')


