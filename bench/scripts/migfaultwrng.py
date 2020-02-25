import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scipy.ndimage import gaussian_filter
import scaas.scaas2dpy as sca2d
from scaas.wavelet import ricker
from scaas.gradtaper import build_taper
from scaas.velocity import create_randomptb,create_randomptb_loc
from resfoc.tpow import tpow
import matplotlib.pyplot as plt
from utils.signal import ampspec1d

# Set up IO
sep = seppy.sep([])

# Read in the model
vaxes,vel = sep.read_file(None,ifname='./dat/vels/velflts/velfltmod0000.H')
vel = vel.reshape(vaxes.n,order='F')
velw = vel[:,:,0].T
# Read in the reflectivity
raxes,ref = sep.read_file(None,ifname='./dat/vels/velflts/velfltref0000.H')
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

# Scale by a random perturbation
nro=10; oro=1.0; dro=0.01
romin = oro - (nro-1)*dro; romax = romin + dro*(2*nro-1)
#rho = create_randomptb(nz,nx,romin,romax,nptsz=1,nptsx=1,octaves=4,period=80,Ngrad=80,persist=0.2,ncpu=10)
rhosm = create_randomptb_loc(nz,nx,romin,romax,350,1024,300,512,
    nptsz=2,nptsx=2,octaves=4,period=80,Ngrad=80,persist=0.2,ncpu=10)
#rho[:100,:] = 1.0
#rhosm = gaussian_filter(rho,sigma=20)
plt.figure(1); plt.imshow(rhosm,cmap='jet')
plt.figure(2); plt.imshow(rvelsm,cmap='jet')
plt.figure(3); plt.imshow(rvelsm*rhosm,cmap='jet')
plt.show()
rvelwr = rvelsm*rhosm

# Save figures
fsize = 24
fig1 = plt.figure(1,figsize=(14,7)); ax1 = fig1.gca()
im1 = ax1.imshow(rvelsm/1000.0,cmap='jet',interpolation='bilinear',extent=[0.0,(nx-1)*dx/1000.0,(nz-1)*dz/1000.0,0.0])
ax1.set_xlabel('X (km)',fontsize=fsize); ax1.set_ylabel('Z (km)',fontsize=fsize); ax1.tick_params(labelsize=fsize)
cbar_ax1 = fig1.add_axes([0.91,0.11,0.02,0.77])
cbar1 = fig1.colorbar(im1,cbar_ax1,format='%.2f')
cbar1.ax.tick_params(labelsize=fsize)
cbar1.set_label('velocity (km/s)',fontsize=fsize)
cbar1.draw_all()
plt.savefig('./fig/migvelsm.png',bbox_inches='tight',dpi=150,transparent=True)
plt.close()

fig2 = plt.figure(2,figsize=(14,7)); ax2 = fig2.gca()
im2 = ax2.imshow(rhosm,cmap='jet',interpolation='bilinear',extent=[0.0,(nx-1)*dx/1000.0,(nz-1)*dz/1000.0,0.0])
ax2.set_xlabel('X (km)',fontsize=fsize); ax2.set_ylabel('Z (km)',fontsize=fsize); ax2.tick_params(labelsize=fsize)
cbar_ax2 = fig2.add_axes([0.91,0.11,0.02,0.77])
cbar2 = fig2.colorbar(im2,cbar_ax2,format='%.2f')
cbar2.ax.tick_params(labelsize=fsize)
cbar2.set_label(r'$\rho$',fontsize=fsize)
cbar2.draw_all()
plt.savefig('./fig/rhosm.png',bbox_inches='tight',dpi=150,transparent=True)
plt.close()

# Set up the acquisition
bx = 25; bz = 25
velp = np.pad(rvel, ((bz,bz),(bx,bx)),'edge').astype('float32')
vmgp = np.pad(rvelwr, ((bz,bz),(bx,bx)),'edge').astype('float32')
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
# Read in the data
daxes,dat = sep.read_file(None,ifname='fltdat.H')
dat = dat.reshape(daxes.n,order='F')
allshot = np.transpose(dat,(2,0,1)).astype('float32')

imgp  = np.zeros([nzp,nxp],dtype='float32')
imgl  = np.zeros([nzp,nxp],dtype='float32')
sca.brnadj(allsrcs,allsrcx,allsrcz,nsrc,allrecx,allrecz,nrec,nsx,vmgp,imgp,allshot,nthreads)
# Apply the laplacian
sca.lapimg(imgp,imgl)

#tap1d,tap = build_taper(nxp,nzp,20,100)
#tapd = imgl*tap
#imgo = tpow(tapd,nzp,0.0,dz,nxp,1.6)

## Write out image
img = imgl[bz+5:nz+bz+5,bx+5:nx+bx+5]
iaxes = seppy.axes([nz,nx],[0.0,0.0],[dz,dx])
sep.write_file(None,iaxes,img,ofname='fltimgwrng3.H')

