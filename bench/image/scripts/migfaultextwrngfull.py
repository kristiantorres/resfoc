import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scipy.ndimage import gaussian_filter
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from scaas.velocity import create_randomptb_loc
from resfoc.tpow import tpow
from resfoc.resmig import preresmig, get_rho_axis
from genutils.plot import plot_wavelet
from genutils.movie import viewimgframeskey
import matplotlib.pyplot as plt

# Set up IO
sep = seppy.sep([])

# Read in the model
vaxes,vel = sep.read_file(None,ifname='../trdat/dat/vels/velflts/velfltmod0000.H')
vel = vel.reshape(vaxes.n,order='F')
velw = vel[:,:,0].T
# Read in the reflectivity
raxes,ref = sep.read_file(None,ifname='../trdat/dat/vels/velflts/velfltref0000.H')
ref = ref.reshape(raxes.n,order='F')
refw = ref[:,:,0].T

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
plt.figure(1); plt.imshow(rvelsm,cmap='jet')
plt.figure(2); plt.imshow(rvelsm*rhosm1*rhosm2,cmap='jet')
plt.figure(3); plt.imshow(rhosm1*rhosm2,cmap='jet')
plt.show()
rvelwr = rvelsm*rhosm1*rhosm2

prp = geom.defaultgeom(nx,dx,nz,dz,nsx=52,dsx=20)

prp.plot_acq(rref,cmap='gray',show=False)

# Create data axes
ntu = 6400; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)
plot_wavelet(wav,dtu)

# Read in the data
daxes,dat = sep.read_file(None,ifname='fltdat.H')
dat = dat.reshape(daxes.n,order='F')
allshot = np.ascontiguousarray(np.transpose(dat,(2,0,1)).astype('float32'))

prp.build_taper(40,130)
prp.plot_taper(rref,cmap='gray')

# Imaging
dtd=0.004
img = prp.wem(rvelwr,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=24)
nh,oh,dh = prp.get_off_axis()

#imgtp = np.zeros(img.shape,dtype='float32')
#for ih in range(nh):
#  imgtp[ih,:,:] = tpow(img[ih,:,:],nz,0.0,dz,nx,1.6)
#viewimgframeskey(imgtp,pclip=0.2,transp=False)
## Write out image
#imgo = np.transpose(imgtp,(1,2,0))
#iaxes = seppy.axes([nz,nx,nh],[0.0,0.0,oh],[dz,dx,dh])
#sep.write_file(None,iaxes,imgo,ofname='fltimgextprcnewwrng.H')

# Residual imaging
inro = 5; idro = 0.005
storm = preresmig(img,[dh,dz,dx],nro=inro,dro=idro,time=False,transp=True,verb=True,nthreads=2)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

#viewimgframeskey(storm[:,int(nh/2),:,:],transp=False,ttlstring='rho=%.3f',ottl=ooro,dttl=odro)

# Convert to angle
stormang = prp.to_angle(storm,nthrds=24,oro=ooro,dro=odro,verb=True)
na,oa,da = prp.get_ang_axis()

#viewimgframeskey(storm[int(onro/2),:,:,:],transp=False,ttlstring='x=%.3f',ottl=0.0,dttl=dx/1000.0)

# Write to file
stormangt = np.transpose(stormang,(3,2,1,0))
aaxes = seppy.axes([nz,na,nx,onro],[0.0,oa,0.0,ooro],[dz,da,dx,odro])
sep.write_file(None,aaxes,stormangt,ofname='stormangwrng.H')

