import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt
from genutils.plot import plot_allanggats

sep = seppy.sep()

# Show data
daxes,dat = sep.read_file("./dat/mymidpts.H")
[nt,nh,nm] = daxes.n; [dt,dh,dm] = daxes.d; [ot,oh,om] = daxes.o
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T)
dat = np.reshape(dat,[nh*nm,nt])
sc = 0.2
dmin = sc*np.min(dat); dmax = sc*np.max(dat)

fsize = 15
fig = plt.figure(figsize=(12,6)); ax = fig.gca()
ax.imshow(dat[0:96].T,cmap='gray',vmin=dmin,vmax=dmax,
           extent=[0,90,nt*dt,ot],aspect=20,interpolation='none')
ax.set_ylabel('Time (s)',fontsize=fsize)
ax.set_xlabel('Trace number',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.savefig('./fig/semfigs/beicmp.png',dpi=150,bbox_inches='tight',transparent=True)

# Show velocity
vaxes,vel = sep.read_file("vintzcomb.H")
[nz,nx] = vaxes.n; [oz,ox] = vaxes.o; [dz,dx] = vaxes.d
vel = vel.reshape(vaxes.n,order='F')

fsize = 16
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
im = ax.imshow(vel,cmap='jet',extent=[ox,ox+nx*dx,oz+nz*dz,oz],interpolation='bilinear',aspect=2.0)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.92,0.15,0.02,0.7])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig('./fig/semfigs/vintz.png',dpi=150,bbox_inches='tight',transparent=True)

# Show migration
aaxes,ang = sep.read_file("spimgbobang.H")
[nz,na,naz,nx] = aaxes.n; [oz,oa,oaz,ox] = aaxes.o; [dz,da,daz,dx] = aaxes.d;
ang = ang.reshape(aaxes.n,order='F').T
angw = ang[20:nx-20,0,32:,:800]
stk = np.sum(angw,axis=1)
sc = 0.4
smin = sc*np.min(stk); smax = sc*np.max(stk)

# Plot the stack
fig = plt.figure(figsize=(12,6)); ax = fig.gca()
ax.imshow(stk.T,cmap='gray',interpolation='bilinear',vmin=smin,vmax=smax,
          extent=[ox,ox+nx*dx,nz*dz,oz],aspect=2.0)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.savefig('./fig/semfigs/mig.png',dpi=150,transparent=True,bbox_inches='tight')

plot_allanggats(angw,dz,dx,jx=8,aagc=False,show=True,pclip=0.5,interp='none',aspect='auto')

# Plot the zero subsurface offset
haxes,hoff = sep.read_file("spimgbobext.H")
[nz,nx,ny,nhx] = haxes.n; [oz,ox,oy,ohx] = haxes.o; [dz,dx,dy,dhx] = haxes.d;
hoff = hoff.reshape(haxes.n,order='F').T
hoffw = hoff[20,0,20:nx-20,:800]
sc = 0.4
smin = sc*np.min(hoffw); smax = sc*np.max(hoffw)

fig = plt.figure(figsize=(12,6)); ax = fig.gca()
ax.imshow(hoffw.T,cmap='gray',interpolation='bilinear',vmin=smin,vmax=smax,
          extent=[ox,ox+nx*dx,nz*dz,oz],aspect=2.0)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.savefig('./fig/semfigs/hoffmig.png',dpi=150,transparent=True,bbox_inches='tight')

# Plot the zero offset image
zaxes,zimg = sep.read_file("zoimg.H")
zimg = zimg.reshape(zaxes.n,order='F')
zimgw = zimg[10:nx-10,0,:800]
sc = 0.4
smin = sc*np.min(zimgw); smax = sc*np.max(zimgw)

fig = plt.figure(figsize=(12,6)); ax = fig.gca()
ax.imshow(zimgw,cmap='gray',interpolation='bilinear',vmin=smin,vmax=smax,
          extent=[ox,ox+nx*dx,nz*dz,oz],aspect=2.0)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
plt.savefig('./fig/semfigs/zomig.png',dpi=150,transparent=True,bbox_inches='tight')


# Show angle gathers


