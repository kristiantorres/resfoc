import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.utils import plotseglabel
from scaas.velocity import salt_mask
from genutils.plot import plot_allanggats, plot_imgvelptb

sep = seppy.sep()

aaxes,ang = sep.read_file("sigsbee_ang.H")
ang  = ang.reshape(aaxes.n,order='F').T
[nz,na,ny,nx] = aaxes.n; [dz,da,dy,dx] = aaxes.d; [oz,oa,oy,ox] = aaxes.o
# Window in x and in angle
bxw = 20; nxw = nx - 20
angw = ang[bxw:nxw,0,32:,:]
stk = np.sum(angw,axis=1)
sc = 0.1
smin = sc*np.min(stk); smax = sc*np.max(stk)
#
## Region of interest
oxw = ox + bxw*dx
xmin = 50; xmax = 200
zmin = 400; zmax=1100

# Plot the stack
#fsize = 15
#fig = plt.figure(figsize=(12,6)); ax = fig.gca()
#ax.imshow(stk.T,cmap='gray',interpolation='bilinear',vmin=smin,vmax=smax,
#          extent=[oxw,oxw+nxw*dx,oz+nz*dz,oz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/semfigs/sigstkimg.png',dpi=150,transparent=True,bbox_inches='tight')
#
## Plot the region of interest
#fig = plt.figure(figsize=(12,6)); ax = fig.gca()
#ax.imshow(stk[xmin:xmax,zmin:zmax].T,cmap='gray',interpolation='bilinear',vmin=smin,vmax=smax,
#          extent=[oxw+xmin*dx,oxw+xmax*dx,zmax*dz,zmin*dz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/semfigs/sigstkimgwind.png',dpi=150,transparent=True,bbox_inches='tight')
#
## Plot the angles
#plot_allanggats(angw,dz,dx,jx=4,aagc=False,figname='./fig/semfigs/sigangs.png',
#                xmin=oxw,pclip=0.1)
#
#plot_allanggats(angw[xmin:xmax,:,zmin:zmax],dz,dx,jx=4,aagc=False,figname='./fig/semfigs/sigangswind.png',
#                zmin=zmin*dz,xmin=oxw+xmin*dx,pclip=0.1,aspect=1.0)

# Plot the velocity error on the image
#eaxes,err = sep.read_file("overwinterp.H")
#err = err.reshape(eaxes.n,order='F').T
#errw = err[bxw:nxw,:]
#
## Plot perturbation on reflectivity
#plot_imgvelptb(stk.T,errw.T,dz,dx,velmin=-100,velmax=100,thresh=5,aagc=False,
#               imin=smin,imax=smax,figname="./fig/semfigs/sigvelptb.png",hbar=0.53,barz=0.23,
#               xmin=oxw,xmax=(oxw+nxw*dx))
#
## Read in image migrated with velocity error
#aaxes,ang = sep.read_file("sigsbee_angoverw.H")
#ang  = ang.reshape(aaxes.n,order='F').T
#[nz,na,ny,nx] = aaxes.n; [dz,da,dy,dx] = aaxes.d; [oz,oa,oy,ox] = aaxes.o
## Window in x and in angle
#angw = ang[bxw:nxw,0,32:,:]
#stk = np.sum(angw,axis=1)
#sc = 0.1
#smin = sc*np.min(stk); smax = sc*np.max(stk)

# Plot the stack
#fsize = 15
#fig = plt.figure(figsize=(12,6)); ax = fig.gca()
#ax.imshow(stk.T,cmap='gray',interpolation='bilinear',vmin=smin,vmax=smax,
#          extent=[oxw,oxw+nxw*dx,oz+nz*dz,oz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/semfigs/sigstkimgwrng.png',dpi=150,transparent=True,bbox_inches='tight')
#
## Plot the region of interest
#fig = plt.figure(figsize=(12,6)); ax = fig.gca()
#ax.imshow(stk[xmin:xmax,zmin:zmax].T,cmap='gray',interpolation='bilinear',vmin=smin,vmax=smax,
#          extent=[oxw+xmin*dx,oxw+xmax*dx,zmax*dz,zmin*dz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/semfigs/sigstkimgwindwrng.png',dpi=150,transparent=True,bbox_inches='tight')
#
## Plot the angles
#plot_allanggats(angw,dz,dx,jx=4,aagc=False,figname='./fig/semfigs/sigangswrng.png',
#                xmin=oxw,pclip=0.1)
#
#plot_allanggats(angw[xmin:xmax,:,zmin:zmax],dz,dx,jx=4,aagc=False,figname='./fig/semfigs/sigangswindwrng.png',
#                zmin=zmin*dz,xmin=oxw+xmin*dx,pclip=0.1,aspect=1.0)


# Read in the stack and the rho from semblance
seaxes,sestk = sep.read_file("stkwind.H")
sestk = np.ascontiguousarray(sestk.reshape(seaxes.n,order='F'))
#seaxes,serho = sep.read_file("sigrho.H")
#serho = np.ascontiguousarray(serho.reshape(seaxes.n,order='F'))
#seaxes,serfi = sep.read_file("sigrfi.H")
#serfi =  np.ascontiguousarray(serfi.reshape(seaxes.n,order='F'))
#
bxw = 20; exw = 480
bzw = 240; ezw = 1150
#
## velocity model to mask the salt
#vaxes,vel = sep.read_file("sigoverw_velint.H")
#vel = vel.reshape(vaxes.n,order='F').T
#velw = vel[bxw:exw,0,bzw:ezw].T
#
nzp = 64; nxp = 64
pe = PatchExtractor((nzp,nxp),stride=(nzp//2,nxp//2))
sestkb = pe.reconstruct(pe.extract(sestk))
#serhob = pe.reconstruct(pe.extract(serho))
#serfib = pe.reconstruct(pe.extract(serfi))
#velb   = pe.reconstruct(pe.extract(velw))
sc2 = 0.2
kmin = sc2*np.min(sestkb); kmax = sc2*np.max(sestkb)
#
## Mask the rho image
#msk,serhomsk = salt_mask(serhob,velb,saltvel=4.5)
#idx = serhomsk == 0.0
#serhomsk[idx] = 1.0
#
## Rho on image
#fsize = 15
#fig = plt.figure(figsize=(10,10)); ax = fig.gca()
#ax.imshow(sestkb,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
#          extent=[ox+bxw*dx,ox+bxw*dx+exw*dx,ezw*dz,bzw*dz+oz])
#im = ax.imshow(serhomsk,cmap='seismic',interpolation='bilinear',vmin=0.975,vmax=1.025,
#               extent=[ox+bxw*dx,ox+bxw*dx+exw*dx,ezw*dz,bzw*dz+oz],alpha=0.2)
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#cbar_ax = fig.add_axes([0.91,0.38,0.02,0.23])
#cbar = fig.colorbar(im,cbar_ax,format='%.2f')
#cbar.solids.set(alpha=1)
#cbar.ax.tick_params(labelsize=fsize)
#cbar.set_label(r'$\rho$',fontsize=fsize)
#plt.savefig("./fig/semfigs/sembrhoimg.png",dpi=150,bbox_inches='tight',transparent=True)
#
## Stack image
#fig = plt.figure(figsize=(10,10)); ax = fig.gca()
#ax.imshow(sestkb,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
#          extent=[ox+bxw*dx,ox+bxw*dx+exw*dx,ezw*dz,bzw*dz+oz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig("./fig/semfigs/sembstk.png",dpi=150,bbox_inches='tight',transparent=True)
#
## Region of interest
#xmin2 = 50; xmax2 = 200
#zmin2 = 160; zmax2=1100
#
#fig = plt.figure(figsize=(12,6)); ax = fig.gca()
#ax.imshow(sestkb[zmin2:zmax2,xmin2:xmax2],interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
#          extent=[oxw+xmin*dx,oxw+xmax*dx,zmax*dz,zmin*dz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig("./fig/semfigs/sembstkwind.png",dpi=150,bbox_inches='tight',transparent=True)
#
## Refocused image
#fig = plt.figure(figsize=(10,10)); ax = fig.gca()
#ax.imshow(serfib,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
#          extent=[ox+bxw*dx,ox+bxw*dx+exw*dx,ezw*dz,bzw*dz+oz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig("./fig/semfigs/sembrfi.png",dpi=150,bbox_inches='tight',transparent=True)
#
#fig = plt.figure(figsize=(12,6)); ax = fig.gca()
#ax.imshow(serfib[zmin2:zmax2,xmin2:xmax2],interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
#          extent=[oxw+xmin*dx,oxw+xmax*dx,zmax*dz,zmin*dz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig("./fig/semfigs/sembrfiwind.png",dpi=150,bbox_inches='tight',transparent=True)
#
## Well-focused image
#stkw = stk[:,bzw:ezw].T
#stkwb = pe.reconstruct(pe.extract(stkw))
#print(stkwb.shape)
#msk,stkwbmsk = salt_mask(stkwb,velb,saltvel=4.5)
#fig = plt.figure(figsize=(10,10)); ax = fig.gca()
#ax.imshow(stkwbmsk,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
#          extent=[ox+bxw*dx,ox+bxw*dx+exw*dx,ezw*dz,bzw*dz+oz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig("./fig/semfigs/wellfoc.png",dpi=150,bbox_inches='tight',transparent=True)
#
#fig = plt.figure(figsize=(12,6)); ax = fig.gca()
#ax.imshow(stkwbmsk[zmin2:zmax2,xmin2:xmax2],interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
#          extent=[oxw+xmin*dx,oxw+xmax*dx,zmax*dz,zmin*dz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig("./fig/semfigs/wellfocwind.png",dpi=150,bbox_inches='tight',transparent=True)

#TODO: one training image plot
# Migration velocity and images with angle gathers
# With velocity error as well
#aaxes,ang = sep.read_file("focex.H")
#ang  = ang.reshape(aaxes.n,order='F').T
#[nz,na,nx] = aaxes.n; [dz,da,dx] = aaxes.d; [oz,oa,ox] = aaxes.o
#
## Window in x and in angle
#bxw = 20; nxw = nx - 20
#angw = ang[bxw:nxw,32:,:]
#stk = np.sum(angw,axis=1)
#sc = 0.1
#smin = sc*np.min(stk); smax = sc*np.max(stk)
#
#fsize = 15
#fig = plt.figure(figsize=(12,6)); ax = fig.gca()
#ax.imshow(stk.T,cmap='gray',interpolation='bilinear',vmin=smin,vmax=smax,
#          extent=[oxw,oxw+nxw*dx,oz+nz*dz,oz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/semfigs/focstkex.png',dpi=150,transparent=True,bbox_inches='tight')
#
## Plot the angles
#plot_allanggats(angw,dz,dx,jx=4,aagc=False,figname='./fig/semfigs/focangs.png',
#                xmin=oxw,pclip=0.1)
#
## Fault labels
#laxes,lbl = sep.read_file("lblex.H")
#lbl = lbl.reshape(laxes.n,order='F')
#lblw = lbl[:,bxw:nxw]
#plotseglabel(stk.T,lblw,fname='./fig/semfigs/foclbl',interp='bilinear',
#             xmin=oxw,xmax=oxw+nxw*dx,zmax=nz*dz,wbox=12,hbox=6,vmin=smin,vmax=smax,
#             labelsize=15,xlabel='X (km)',ylabel='Z (km)',ticksize=15)
#
## Defocused example
#aaxes,ang = sep.read_file("defex.H")
#ang  = ang.reshape(aaxes.n,order='F').T
#[nz,na,nx] = aaxes.n; [dz,da,dx] = aaxes.d; [oz,oa,ox] = aaxes.o
#angw = ang[bxw:nxw,32:,:]
#stk = np.sum(angw,axis=1)
#
#fsize = 15
#fig = plt.figure(figsize=(12,6)); ax = fig.gca()
#ax.imshow(stk.T,cmap='gray',interpolation='bilinear',vmin=smin,vmax=smax,
#          extent=[oxw,oxw+nxw*dx,oz+nz*dz,oz])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/semfigs/defstkex.png',dpi=150,transparent=True,bbox_inches='tight')
#
## Plot the angles
#plot_allanggats(angw,dz,dx,jx=4,aagc=False,figname='./fig/semfigs/defangs.png',
#                xmin=oxw,pclip=0.1)

# CNN estimation of Rho
# Read in the stack and the rho from semblance
seaxes,sestk = sep.read_file("stkfocwind3.H")
sestk = np.ascontiguousarray(sestk.reshape(seaxes.n,order='F'))
seaxes,serho = sep.read_file("sigfocrho3.H")
serho = np.ascontiguousarray(serho.reshape(seaxes.n,order='F'))
seaxes,serfi = sep.read_file("sigfocrfi3.H")
serfi =  np.ascontiguousarray(serfi.reshape(seaxes.n,order='F'))

# Rho on image
fsize = 15
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(sestkb,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[ox+bxw*dx,ox+bxw*dx+exw*dx,ezw*dz,bzw*dz+oz])
im = ax.imshow(serho,cmap='seismic',interpolation='bilinear',vmin=0.975,vmax=1.025,
               extent=[ox+bxw*dx,ox+bxw*dx+exw*dx,ezw*dz,bzw*dz+oz],alpha=0.2)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.91,0.38,0.02,0.23])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.solids.set(alpha=1)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r'$\rho$',fontsize=fsize)
plt.savefig("./fig/semfigs/cnnrhoimg.png",dpi=150,bbox_inches='tight',transparent=True)

# Stack image
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(sestk,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[ox+bxw*dx,ox+bxw*dx+exw*dx,ezw*dz,bzw*dz+oz])
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.savefig("./fig/semfigs/cnnstk.png",dpi=150,bbox_inches='tight',transparent=True)

# Region of interest
xmin2 = 50; xmax2 = 200
zmin2 = 160; zmax2=1100

fig = plt.figure(figsize=(12,6)); ax = fig.gca()
ax.imshow(sestk[zmin2:zmax2,xmin2:xmax2],interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[oxw+xmin*dx,oxw+xmax*dx,zmax*dz,zmin*dz])
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.savefig("./fig/semfigs/cnnstkwind.png",dpi=150,bbox_inches='tight',transparent=True)

## Refocused image
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(serfi,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[ox+bxw*dx,ox+bxw*dx+exw*dx,ezw*dz,bzw*dz+oz])
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.savefig("./fig/semfigs/cnnrfi.png",dpi=150,bbox_inches='tight',transparent=True)

fig = plt.figure(figsize=(12,6)); ax = fig.gca()
ax.imshow(serfi[zmin2:zmax2,xmin2:xmax2],interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[oxw+xmin*dx,oxw+xmax*dx,zmax*dz,zmin*dz])
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.savefig("./fig/semfigs/cnnrfiwind.png",dpi=150,bbox_inches='tight',transparent=True)

