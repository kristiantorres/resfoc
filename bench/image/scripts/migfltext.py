import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scaas.trismooth import smooth
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from resfoc.gain import agc,tpow
from resfoc.resmig import preresmig, get_rho_axis
import matplotlib.pyplot as plt
from genutils.plot import plot_wavelet
from genutils.movie import viewimgframeskey

# Create SEP IO object
sep = seppy.sep()

# Read in the model
vaxes,vel = sep.read_file('../fltdat/velbig.H')
vel = vel.reshape(vaxes.n,order='F')
velw = np.ascontiguousarray(vel.astype('float32'))
# Read in the reflectivity
raxes,ref = sep.read_file('../fltdat/refbig.H')
ref = ref.reshape(raxes.n,order='F')
refw = np.ascontiguousarray(ref.astype('float32'))

nz = vaxes.n[0]; dz = vaxes.d[0]
nx = vaxes.n[1]; dx = vaxes.d[1]

# Create migration velocity
rvelsm = smooth(velw,rect1=30,rect2=30)

dsx = 20; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=51,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(rvelsm,cmap='jet',show=False)
prp.plot_acq(refw,cmap='gray',show=False)

# Create data axes
ntu = 6500; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)
plot_wavelet(wav,dtu)

# Read in the data
daxes,dat = sep.read_file('fltbig.H')
dat = dat.reshape(daxes.n,order='F')
allshot = np.ascontiguousarray(np.transpose(dat,(2,0,1)).astype('float32'))
dtd = daxes.d[0]

prp.build_taper(70,150)
prp.plot_taper(refw,cmap='gray')

img = prp.wem(rvelsm,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=24)
nh,oh,dh = prp.get_off_axis()

# Write out image
imgo = np.transpose(img,(1,2,0))
sep.write_file('fltimgbig.H',imgo,os=[0.0,0.0,oh],ds=[dz,dx,dh])
#sep.write_file('fltimgbig.H',img,ds=[dz,dx])

#inro = 17; idro = 0.00125
#storm = preresmig(img,[dh,dz,dx],nro=inro,dro=idro,time=True,transp=True,verb=True,nthreads=24)
#onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

#stormang = prp.to_angle(storm,nthrds=24,oro=ooro,dro=odro,verb=True)
#na,oa,da = prp.get_ang_axis()

# Write to file
#stormangt = np.transpose(stormang,(3,2,1,0))
#aaxes = seppy.axes([nz,na,nx,onro],[0.0,oa,0.0,ooro],[dz,da,dx,odro])
#sep.write_file("fltang.H",stormangt,ofaxes=aaxes)

#sep.to_header("rugsmall.H","srcz=%d recz=%d"%(0,0))
#sep.to_header("rugsmall.H","bx=%d bz=%d alpha=%f"%(bx,bz,0.99))

