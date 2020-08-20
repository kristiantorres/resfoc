import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scipy.ndimage import gaussian_filter
import scaas.defaultgeom as geom
from scaas.velocity import create_randomptb_loc
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

# Scale by a random perturbation
nro1=3; oro1=1.03; dro1=0.01
romin1 = oro1 - (nro1-1)*dro1; romax1 = romin1 + dro1*(2*nro1-1)
rhosm1 = create_randomptb_loc(nz,nx,romin1,romax1,150,150,200,700,
    nptsz=2,nptsx=2,octaves=3,period=80,Ngrad=80,persist=0.2,ncpu=1)

nro2=3; oro2=0.97; dro2=0.01
romin2 = oro2 - (nro2-1)*dro1; romax2 = romin2 + dro2*(2*nro2-1)
rhosm2 = create_randomptb_loc(nz,nx,romin2,romax2,150,150,120,300,
    nptsz=2,nptsx=2,octaves=3,period=80,Ngrad=80,persist=0.2,ncpu=1)
# Plot for QC
plt.figure(1); plt.imshow(rvelsm,cmap='jet')
plt.figure(2); plt.imshow(rvelsm*rhosm1*rhosm2,cmap='jet')
plt.figure(3); plt.imshow(rhosm1*rhosm2,cmap='jet')
plt.show()
rvelwr = rvelsm*rhosm1*rhosm2

dsx = 20; bx = 25; bz = 25
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=52,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(rvelwr,cmap='jet',show=False)
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

img = prp.wem(rvelwr,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=24)
nh,oh,dh = prp.get_off_axis()

# Residual migration
inro = 10; idro = 0.0025
storm = preresmig(img,[dh,dz,dx],nps=[2049,513,1025],nro=inro,dro=idro,time=tconv,transp=True,verb=True,nthreads=19)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

if(aconv):
  # Convert to angle
  stormang = prp.to_angle(storm,oro=ooro,dro=odro,verb=True,nthrds=24)
  na,oa,da = prp.get_angle_axis()
  # Write to file
  stormangt = np.transpose(stormang,(2,1,3,0))
  if(tconv): dz = dtd
  sep.write_file("resangwrng.H",stormangt,os=[0,oa,0,ooro],ds=[dz,da,dx,odro])
else:
  # Write to file
  stormt = np.transpose(storm,(2,3,1,0))
  if(tconv): dz = dtd
  sep.write_file("resangwrng.H",stormt,os=[0,0,oh,ooro],ds=[dz,dx,dh,odro])

