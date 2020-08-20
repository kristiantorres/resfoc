import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scaas.trismooth import smooth
import scaas.defaultgeom as geom
from scaas.velocity import create_randomptbs_loc
from scaas.wavelet import ricker
from resfoc.gain import agc,tpow
from resfoc.resmig import preresmig, get_rho_axis
import matplotlib.pyplot as plt
from genutils.plot import plot_wavelet, plot_imgvelptb
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

# Scale by a random perturbation
rho = create_randomptbs_loc(nz,nx,nptbs=3,romin=0.95,romax=1.00,
                            minnaz=100,maxnaz=150,minnax=100,maxnax=400,mincz=100,maxcz=150,mincx=250,maxcx=700,
                            mindist=100,nptsz=2,nptsx=2,octaves=2,period=80,persist=0.2,ncpu=1,sigma=20)

# Read in the random perturbation
#paxes,velptb= sep.read_file('velptbbig.H')
#velptb = velptb.reshape(paxes.n,order='F')
#velptb = np.ascontiguousarray(velptb.astype('float32'))

rvelwr = rvelsm*rho
velptb = rvelwr - rvelsm
#rvelwr = rvelsm + velptb
plot_imgvelptb(refw,velptb,dz,dx,velmin=-100,velmax=100,thresh=5,agc=False,show=True)

dsx = 20; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=51,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(rvelwr,cmap='jet',show=False)
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

img = prp.wem(rvelwr,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=24)
nh,oh,dh = prp.get_off_axis()

# Write out image
imgo = np.transpose(img,(1,2,0))
sep.write_file('fltimgbigwrng.H',imgo,os=[0.0,0.0,oh],ds=[dz,dx,dh])
#sep.write_file('fltimgbigwrng.H',img,ds=[dz,dx])
sep.write_file('velptbbig.H',velptb,ds=[dz,dx])

#inro = 5; idro = 0.005
#storm = preresmig(img,[dh,dz,dx],nro=inro,dro=idro,time=False,transp=True,verb=True,nthreads=2)
#onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

#stormang = prp.to_angle(storm,nthrds=4,oro=ooro,dro=odro,verb=True)
#na,oa,da = prp.get_ang_axis()

# Write to file
#stormangt = np.transpose(stormang,(3,2,1,0))
#aaxes = seppy.axes([nz,na,nx,onro],[0.0,oa,0.0,ooro],[dz,da,dx,odro])
#sep.write_file("stormang.H",stormangt,ofaxes=aaxes)

#sep.to_header("rugsmall.H","srcz=%d recz=%d"%(0,0))
#sep.to_header("rugsmall.H","bx=%d bz=%d alpha=%f"%(bx,bz,0.99))

