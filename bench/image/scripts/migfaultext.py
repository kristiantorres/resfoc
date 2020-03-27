import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scipy.ndimage import gaussian_filter
import scaas.scaas2dpy as sca2d
from scaas.wavelet import ricker
from scaas.gradtaper import build_taper
from resfoc.tpow import tpow
import matplotlib.pyplot as plt
from utils.signal import ampspec1d

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
nx = 1024; nz=512
rvel = (resample(velw,[nx,nz],kind='linear')).T
rref = (resample(refw,[nx,nz],kind='linear')).T
#dz = vaxes.d[0]; dx = vaxes.d[1]/2.0
dz = 10; dx = 10

# Create migration velocity
rvelsm = gaussian_filter(rvel,sigma=20)
dvel =  rvel - rvelsm

# Set up the acquisition
bx = 25; bz = 25
velp = np.pad(rvel, ((bz,bz),(bx,bx)),'edge').astype('float32')
vmgp = np.pad(rvelsm, ((bz,bz),(bx,bx)),'edge').astype('float32')
#dvlp = np.pad(dvel, ((bz,bz),(bx,bx)),'edge').astype('float32')
dvlp = np.pad(rref, ((bz,bz),(bx,bx)),'edge').astype('float32')
nzp,nxp = velp.shape

# Source coordinates
nsx = 52; dsx = 20; osx = 0.0; srcz = 0.0
osxp  = osx  + bx + 5
srczp = srcz + bz + 5
nsrc = np.ones(nsx,dtype='int32')
allsrcx = np.zeros([nsx,1],dtype='int32')
allsrcz = np.zeros([nsx,1],dtype='int32')
srcs = np.linspace(osxp,osxp + (nsx-1)*dsx,nsx)
for isx in range(nsx):
  allsrcx[isx,0] = int(srcs[isx])
  allsrcz[isx,0] = int(srczp)

# Create receiver coordinates
orx = 0.0; recz = 0.0
nrx = nx; orxp = bx + 5; drx = 1
reczp = bz + 5
nrec = np.zeros(nrx,dtype='int32') + nrx
allrecx = np.zeros([nsx,nrx],dtype='int32')
allrecz = np.zeros([nsx,nrx],dtype='int32')
# Create all receiver positions
recs = np.linspace(orxp,orxp + (nrx-1)*drx,nrx)
for isx in range(nsx):
  allrecx[isx,:] = (recs[:]).astype('int32')
  allrecz[isx,:] = np.zeros(len(recs),dtype='int32') + reczp

# Plot acquisition
plt.figure(1)
# Plot velocity model
vmin = np.min(velp); vmax = np.max(velp)
#plt.imshow(velp,extent=[0,nxp,nzp,0],vmin=vmin,vmax=vmax,cmap='jet')
plt.imshow(dvlp,extent=[0,nxp,nzp,0],cmap='gray')
# Get all source positions
plt.scatter(allrecx[0,:],allrecz[0,:])
plt.scatter(allsrcx[:,0],allsrcz[:,0])
plt.show()

# Create data axes
ntu = 6400; otu = 0.0; dtu = 0.001;
ntd = 1600; otd = 0.0; dtd = 0.004;

# Create the wavelet array
freq = 20; amp = 100.0; dly = 0.2;
fsrc = ricker(ntu,dtu,freq,amp,dly)
spec,fs = ampspec1d(fsrc,dtu)
plt.plot(fs,spec); plt.show()
allsrcs = np.zeros([nsx,1,ntu],dtype='float32')
for isx in range(nsx):
  allsrcs[isx,0,:] = fsrc[:]

# Create output data array
fact = int(dtd/dtu);
allshot = np.zeros((nsx,ntd,nrx),dtype='float32')

# Set up a wave propagation object
alpha=0.99
sca = sca2d.scaas2d(ntd,nxp,nzp,dtd,dx,dz,dtu,bx,bz,alpha=alpha)

nthreads=24
#sca.fwdprop_multishot(allsrcs,allsrcx,allsrcz,nsrc,allrecx,allrecz,nrec,nsx,velp,allshot,nthreads)
#sca.brnfwd(allsrcs,allsrcx,allsrcz,nsrc,allrecx,allrecz,nrec,nsx,vmgp,dvlp,allshot,nthreads,verb=1)
# Read in the data
daxes,dat = sep.read_file(None,ifname='fltdat.H')
dat = dat.reshape(daxes.n,order='F')
allshot = np.transpose(dat,(2,0,1)).astype('float32')

tap1d,tap = build_taper(nxp,nzp,20,100)

nh = 0; rnh = 2*nh + 1; zoff = nh
imgp  = np.zeros([rnh,nzp,nxp],dtype='float32')
imgl  = np.zeros([rnh,nzp,nxp],dtype='float32')
plt.figure(); plt.imshow(vmgp,cmap='jet'); plt.show()
print(vmgp.shape)
sca.brnoffadj(allsrcs,allsrcx,allsrcz,nsrc,allrecx,allrecz,nrec,nsx,vmgp,rnh,imgp,allshot,nthreads)
plt.figure()
plt.imshow(imgp[nh,:,:],cmap='gray'); plt.colorbar()
plt.show()
# Apply the laplacian for all offsets
for ih in range(rnh):
  sca.lapimg(imgp[ih,:,:],imgl[ih,:,:])
  imgl[ih,:,:] *= tap
  #TODO: replace this with AGC
  imgl[ih,:,:] = tpow(imgl[ih,:,:],nzp,0.0,dz,nxp,1.6)

## Write out all shots
#datout = np.transpose(allshot,(1,2,0))
#daxes = seppy.axes([ntd,nrx,nsx],[0.0,orx,osx],[dtd,drx,dsx])
#sep.write_file(None,daxes,datout,ofname='fltdat.H')

## Write out image
img = np.transpose(imgl[:,bz+5:nz+bz+5,bx+5:nx+bx+5],(1,2,0))
iaxes = seppy.axes([nz,nx,rnh],[0.0,0.0,-dx*nh],[dz,dx,dx])
sep.write_file(None,iaxes,img,ofname='fltimgextprc1.H')

## Write auxiliary information to header
#sep.to_header(None,"srcz=%d recz=%d"%(srcz,recz),ofname='fltdat.H')
#sep.to_header(None,"bx=%d bz=%d alpha=%f"%(bx,bz,alpha),ofname='fltdat.H')

