import inpout.seppy as seppy
from scaas.velocity import insert_circle
import scaas.defaultgeom as geom
import numpy as np
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey
from genutils.ptyprint import create_inttag

sep = seppy.sep()

nz,nx = 512,1024
dz,dx = 10,10
vel = np.zeros([nz,nx],dtype='float32') + 3000

velcirc = insert_circle(vel,dz,dx,centerx=5120,centerz=2560,rad=15,val=2500)

nsx = 52; dsx = 20; bx = 100; bz = 100
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=nsx,dsx=dsx,bx=bx,bz=bz)
#prp.plot_acq(velcirc,cmap='jet',shpw=True)

# Read in the data
daxes,dat = sep.read_file("intro_dat.H")
dat = dat.reshape(daxes.n,order='F')

# Read in the wavefield
waxes,wfld = sep.read_file("intro_wfld.H")
wfld = wfld.reshape(waxes.n,order='F').T
[nxp,nzp,ntw] = waxes.n; [dx,dz,dtw] = waxes.d
dx /= 1000.0; dz /= 1000.0

# Window the wavefield
wwfld = wfld[:,100:nzp-100,100:nxp-100]
sc1 = 0.05
wmin = sc1*np.min(wwfld); wmax = sc1*np.max(wwfld)
#viewimgframeskey(wwfld,transp=False)

# Make receivers
recx = (prp.allrecx[0]-100)*dx; recz = (prp.allrecz[0]-100)*dz

# Make source position
srcx = (prp.allsrcx[26]-100)*dx; srcz = (prp.allsrcz[26]-100)*dz
srcsx = (prp.allsrcx-100)*dx; srcsz = (prp.allsrcz-100)*dz
srcsxw = 4*dx + srcsx[:-1]; srcszw = 4*dx + srcsz[:-1]

# Just plot velocity model
fsize = 15
#fig = plt.figure(figsize=(10,5)); ax = fig.gca()
#im = ax.imshow(velcirc/1000.0,cmap='jet',extent=[0,nx*dx,nz*dz,0])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#cbar_ax = fig.add_axes([0.91,0.15,0.02,0.7])
#cbar = fig.colorbar(im,cbar_ax,format='%.2f')
#cbar.ax.tick_params(labelsize=fsize)
#cbar.set_label('Velocity (km/s)',fontsize=fsize)
#plt.savefig("./fig/velcirc.png",dpi=150,bbox_inches='tight',transparent=True)
#
## Plot receivers on model
recxw = 4*dx + recx[:-4]; reczw = recz[:-4]
#fig = plt.figure(figsize=(10,5)); ax = fig.gca()
#im = ax.imshow(velcirc/1000.0,cmap='jet',extent=[0,nx*dx,nz*dz,0])
#ax.scatter(recxw[::20],reczw[::20],c='tab:green',marker='v',s=25)
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#cbar_ax = fig.add_axes([0.91,0.15,0.02,0.7])
#cbar = fig.colorbar(im,cbar_ax,format='%.2f')
#cbar.ax.tick_params(labelsize=fsize)
#cbar.set_label('Velocity (km/s)',fontsize=fsize)
#plt.savefig("./fig/velcircrec.png",dpi=150,bbox_inches='tight',transparent=True)
#
## Plot source on receivers
#fig = plt.figure(figsize=(10,5)); ax = fig.gca()
#im = ax.imshow(velcirc/1000.0,cmap='jet',extent=[0,nx*dx,nz*dz,0])
#ax.scatter(recxw[::20],reczw[::20],c='tab:green',marker='v',s=25)
#ax.scatter(srcx+4*dx,srcz+4*dx,c='tab:red',marker='*',s=150)
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#cbar_ax = fig.add_axes([0.91,0.15,0.02,0.7])
#cbar = fig.colorbar(im,cbar_ax,format='%.2f')
#cbar.ax.tick_params(labelsize=fsize)
#cbar.set_label('Velocity (km/s)',fontsize=fsize)
#plt.savefig("./fig/velcircrecsrc.png",dpi=150,bbox_inches='tight',transparent=True)

# Plot all sources on receivers
#fig = plt.figure(figsize=(10,5)); ax = fig.gca()
#im = ax.imshow(velcirc/1000.0,cmap='jet',extent=[0,nx*dx,nz*dz,0])
#ax.scatter(recxw[::20],reczw[::20],c='tab:green',marker='v',s=25)
#ax.scatter(srcsxw[::2],srcszw[::2],c='tab:red',marker='*',s=150)
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#cbar_ax = fig.add_axes([0.91,0.15,0.02,0.7])
#cbar = fig.colorbar(im,cbar_ax,format='%.2f')
#cbar.ax.tick_params(labelsize=fsize)
#cbar.set_label('Velocity (km/s)',fontsize=fsize)
#plt.savefig("./fig/velcircrecsrcs.png",dpi=150,bbox_inches='tight',transparent=True)

# Plot all sources on receivers
velconst = np.zeros(velcirc.shape,dtype='float32') + 3.0
fig = plt.figure(figsize=(10,5)); ax = fig.gca()
im = ax.imshow(velconst,cmap='jet',extent=[0,nx*dx,nz*dz,0],vmin=2.5,vmax=3.0)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.91,0.15,0.02,0.7])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig("./fig/velmig.png",dpi=150,bbox_inches='tight',transparent=True)

#fdat = np.zeros([ntw,nxp-200],dtype='float32')
#fsize = 15
#k = 0
#for it in range(0,ntw):
#  f,axarr = plt.subplots(1,2,figsize=(15,10),gridspec_kw={'width_ratios':[2,1]})
#  axarr[0].scatter(recx[::20],recz[::20],c='tab:green',marker='v',s=25)
#  axarr[0].scatter(srcx,srcz,c='tab:red',marker='*',s=150)
#  im = axarr[0].imshow(wwfld[it],extent=[0,nx*dx,nz*dz,0],cmap='gray',interpolation='sinc',
#                       vmin=wmin,vmax=wmax)
#  axarr[0].imshow(velcirc,cmap='jet',extent=[0,nx*dx,nz*dz,0],alpha=0.5)
#  axarr[0].set_xlabel('X (km)',fontsize=fsize)
#  axarr[0].set_ylabel('Z (km)',fontsize=fsize)
#  axarr[0].tick_params(labelsize=fsize)
#  # Plot the data
#  fdat[it,:] = wwfld[it,5,:]
#  axarr[1].imshow(fdat,cmap='gray',interpolation='sinc',extent=[0,nx*dx,ntw*dtw,0],vmin=wmin,vmax=wmax,aspect=4.0)
#  axarr[1].set_xlabel('X (km)',fontsize=fsize)
#  axarr[1].set_ylabel('Time (s)',fontsize=fsize)
#  axarr[1].tick_params(labelsize=fsize)
#  if(it%4 == 0):
#    plt.savefig('./fig/wfldmovie/frame-%d.png'%(k),bbox_inches='tight',transparent=True,dpi=150)
#    #plt.show()
#    k += 1

