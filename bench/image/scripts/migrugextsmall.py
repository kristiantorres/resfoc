import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scaas.trismooth import smooth
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from resfoc.gain import agc,tpow
from resfoc.resmig import preresmig, get_rho_axis
import matplotlib.pyplot as plt
from utils.plot import plot_wavelet
from utils.movie import viewimgframeskey

# Create SEP IO object
sep = seppy.sep()

# Read in the model
vaxes,vel = sep.read_file('velrug.H')
vel = vel.reshape(vaxes.n,order='F')
velw = np.ascontiguousarray(vel.astype('float32'))
# Read in the reflectivity
raxes,ref = sep.read_file('lyrrug.H')
ref = ref.reshape(raxes.n,order='F')
refw = np.ascontiguousarray(ref.astype('float32'))

nz = vaxes.n[0]; dz = vaxes.d[0] 
nx = vaxes.n[1]; dx = vaxes.d[1]

# Create migration velocity
rvelsm = smooth(velw,rect1=20,rect2=20)

dsx = 5; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=52,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(rvelsm,cmap='jet',show=False)
prp.plot_acq(refw,cmap='gray',show=False,vmin=-1,vmax=1)

# Create data axes
ntu = 3000; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)
plot_wavelet(wav,dtu)

dtd = 0.004
#allshot = prp.model_lindata(rvelsm,refw,wav,dtd,verb=True,nthrds=4)

#viewimgframeskey(allshot,transp=False,pclip=0.2)

# Read in the data
daxes,dat = sep.read_file('rugsmall.H')
dat = dat.reshape(daxes.n,order='F')
allshot = np.ascontiguousarray(np.transpose(dat,(2,0,1)).astype('float32'))

prp.build_taper(50,100)
prp.plot_taper(refw,cmap='gray',vmin=-1,vmax=1)

img = prp.wem(rvelsm,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=4)
nh,oh,dh = prp.get_off_axis()

#imgtp = np.zeros(img.shape,dtype='float32')
#for ih in range(nh):
#  imgtp[ih,:,:] = tpow(img[ih,:,:],nz,0.0,dz,nx,1.6)
#viewimgframeskey(imgtp,pclip=0.2,transp=False)
# Write out image
#imgo = np.transpose(img,(1,2,0))
#sep.write_file('rugimgsmall.H',imgo,ors=[0.0,0.0,oh],ds=[dz,dx,dh])

inro = 5; idro = 0.005
storm = preresmig(img,[dh,dz,dx],nro=inro,dro=idro,time=False,transp=True,verb=True,nthreads=2)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

stormang = prp.to_angle(storm,nthrds=4,oro=ooro,dro=odro,verb=True)
na,oa,da = prp.get_ang_axis()

# Write to file
stormangt = np.transpose(stormang,(3,2,1,0))
aaxes = seppy.axes([nz,na,nx,onro],[0.0,oa,0.0,ooro],[dz,da,dx,odro])
sep.write_file("stormang.H",stormangt,ofaxes=aaxes)

## Write out all shots
#datout = np.transpose(allshot,(1,2,0))
#sep.write_file('rugsmall.H',datout,ds=[dtd,dx,dsx])

#sep.to_header("rugsmall.H","srcz=%d recz=%d"%(0,0))
#sep.to_header("rugsmall.H","bx=%d bz=%d alpha=%f"%(bx,bz,0.99))

