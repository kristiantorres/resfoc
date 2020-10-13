import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from oway.utils import interp_vel
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in RMS velocity
raxes,vrms = sep.read_file("velrmscomb.H")
otvr,oxvr = raxes.o; dtvr,dxvr = raxes.d; ntvr,nxvr = raxes.n
vrms = vrms.reshape(raxes.n,order='F')

# Read in Interval velocity in time
taxes,vintt = sep.read_file("vinttcomb.H")
otvit,oxvit = taxes.o; dtvit,dxvit = taxes.d; ntvit,nxvit = taxes.n
vintt = vintt.reshape(taxes.n,order='F')

# Read in interval velocity in depth
zaxes,vintz = sep.read_file("vintzcomb.H")
ozviz,oxviz = zaxes.o; dzviz,dxviz = zaxes.d; nzviz,nxviz = zaxes.n
vintz = vintz.reshape(zaxes.n,order='F')

# Read in zero offset image
oaxes,zoimg = sep.read_file("zoimg.H")
ooz,ooy,oox = oaxes.o; doz,doy,dox = oaxes.d; noz,noy,nox = oaxes.n
zoimg = zoimg.reshape(oaxes.n,order='F').T

# Read in subsurface offset image
paxes,spimgoff = sep.read_file("spimgextbobdistr.H")
nzp,nxp,nyp,nhx = paxes.n; ozp,oxp,oyp,ohxp = paxes.o; dzp,dxp,dyp,dhxp = paxes.o
spimgoff= spimgoff.reshape(paxes.n,order='F').T

# Read in subsurface angle image
aaxes,spimgang = sep.read_file("spimgbobang.H")
nzp,na,nyp,nxp = aaxes.n; ozp,oa,oyp,oxp = aaxes.o; dzp,da,dyp,dxp = aaxes.d
spimgang = spimgang.reshape(aaxes.n,order='F').T

# Interpolate the RMS velocity
vrmsin = np.zeros([nzp,1,nxvr],dtype='float32')
vrmsin[:,0,:] = vrms[:nzp,:]
vrmsi = interp_vel(nzp,nyp,oyp,dyp,nxp,oxp,dxp,
                   vrmsin,dxvr,1.0,ovx=oxvr,ovy=0.0)

# Interpolate the interval velocity in time
vinttin = np.zeros([nzp,1,nxvit],dtype='float32')
vinttin[:,0,:] = vintt[:nzp,:]
vintti = interp_vel(nzp,nyp,oyp,dyp,nxp,oxp,dxp,
                    vinttin,dxvit,1.0,ovx=oxvit,ovy=0.0)

# Interpolate the interval velocity in depth
vintzin = np.zeros([nzp,1,nxviz],dtype='float32')
vintzin[:,0,:] = vintz[:nzp,:]
vintzi = interp_vel(nzp,nyp,oyp,dyp,nxp,oxp,dxp,
                    vintzin,dxvit,1.0,ovx=oxviz,ovy=0.0)

# Windowing
begx = 25; endx = nxp-25; nxw = endx - begx
begz = 0;  endz = 800;    nzw = endz - begz

pxb = begx*dxp; pxe = begx*dxp + nxw*dxp
pzz = begz*dzp;  pze = begz*dzp + nzw*dzp

# Get the zero subsurface offset image
zsso = spimgoff[20,0,:,:]

# Get the angle stack
astk = np.sum(spimgang,axis=2)[:,0,:]

# Plotting parameters
velmin = 1.45; velmax = 3.1
sc = 0.4
omin = np.min(zsso);  omax = np.max(zsso)
amin = np.min(astk);  amax = np.max(astk)
zmin = np.min(zoimg); zmax = np.max(zoimg)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'AVHershey Simplex'
fsize = 15; figsize=(10,6)
taspect = 2.5; zaspect = 2.0

# RMS velocity
fig = plt.figure(figsize=figsize); ax = fig.gca()
imv = ax.imshow(vrmsi[begz:endz,0,begx:endx],cmap='jet',vmin=velmin,vmax=velmax,
          interpolation='bilinear',extent=[pxb,pxe,endz*dtvr,0],aspect=taspect)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Time (s)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.79,0.12,0.02,0.75])
cbar = fig.colorbar(imv,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig('./fig/biondo/rmsvel_t.pdf',dpi=150,transparent=True,bbox_inches='tight')
plt.close()

# Interval velocity in time
fig = plt.figure(figsize=figsize); ax = fig.gca()
im = ax.imshow(vintti[begz:endz,0,begx:endx],cmap='jet',vmin=velmin,vmax=velmax,
          interpolation='bilinear',extent=[pxb,pxe,endz*dtvit,0],aspect=taspect)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Time (s)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.79,0.12,0.02,0.75])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig('./fig/biondo/intvel_t.pdf',dpi=150,transparent=True,bbox_inches='tight')
plt.close()

# Interval velocity in depth
fig = plt.figure(figsize=figsize); ax = fig.gca()
im = ax.imshow(vintzi[begx:endx,0,begz:endz],cmap='jet',vmin=velmin,vmax=velmax,
              interpolation='bilinear',extent=[pxb,pxe,endz*dzviz,0],aspect=zaspect)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.79,0.12,0.02,0.75])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig('./fig/biondo/intvel_z.pdf',dpi=150,transparent=True,bbox_inches='tight')
plt.close()

# Zero sub-surface offset image
fig = plt.figure(figsize=figsize); ax = fig.gca()
im = ax.imshow(zsso[begx:endx,begz:endz].T,cmap='gray',interpolation='bilinear',
               vmin=sc*omin,vmax=sc*omax,extent=[pxb,pxe,endz*dzp,0],aspect=zaspect)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.79,0.12,0.02,0.75])
cbar = fig.colorbar(imv,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig('./fig/biondo/prestack_off.pdf',dpi=150,transparent=True,bbox_inches='tight')
plt.close()

# Zero offset image
fig = plt.figure(figsize=figsize); ax = fig.gca()
im = ax.imshow(zoimg[10:nox-10,0,0:endz].T,cmap='gray',interpolation='bilinear',
               vmin=0.3*zmin,vmax=0.3*zmax,extent=[pxb,pxe,endz*dzp,0],aspect=zaspect)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.79,0.12,0.02,0.75])
cbar = fig.colorbar(imv,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig('./fig/biondo/zoimg.pdf',dpi=150,transparent=True,bbox_inches='tight')
plt.close()

# Angle stack image
fig = plt.figure(figsize=figsize); ax = fig.gca()
im = ax.imshow(astk[begx:endx,begz:endz].T,cmap='gray',interpolation='bilinear',
               vmin=0.5*amin,vmax=0.5*amax,extent=[pxb,pxe,endz*dzp,0],aspect=zaspect)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.79,0.12,0.02,0.75])
cbar = fig.colorbar(imv,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig('./fig/biondo/prestack_ang.pdf',dpi=150,transparent=True,bbox_inches='tight')
plt.close()

