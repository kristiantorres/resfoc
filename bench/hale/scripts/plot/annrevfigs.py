import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
from genutils.plot import plot_img2d, plot_imgvelptb, plot_vel2d, plot_rhoimg2d
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Read in the image
sep = seppy.sep()
iaxes,img = sep.read_file("spimgbobangwrng.H")
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)[:,0,:,:]
#stkw = agc(np.sum(img,axis=1))[30:542,:512]
stkw = np.sum(img,axis=1)[30:542,:512]
dz,da,dy,dx = iaxes.d; oz,oa,oy,ox = iaxes.o
nx,nz = stkw.shape
pclip=0.6
vmin,vmax = np.min(stkw), np.max(stkw)

# Read in the refocused image
faxes,foc = sep.read_file("realtorch_rfi.H")
foc = foc.reshape(faxes.n,order='F').T

# Read in estimated rho
raxes,rho = sep.read_file("realtorch_rho.H")
rho = rho.reshape(raxes.n,order='F').T

plot_rhoimg2d(stkw[:,100:356].T,rho.T,dz=dz,dx=dx,oz=100*dz,ox=ox,aspect=3.0,
              hbar=0.7,figname='./fig/annrev/estrho.png')

# Read in synthetic image
saxes,syn = sep.read_file("wemvaimg.H")
syn = syn.reshape(saxes.n,order='F')[:,0,:].T
plot_img2d(syn[140:652,:512].T,dz=dz,dx=dx,ox=ox,aspect=2.0,figname='./fig/annrev/synimg.png')

# Velocity model
vaxes,ivel = sep.read_wind("hale_trvels.H",nw=1,fw=0)
nz,nx = vaxes.n; dz,dx,dm = vaxes.d; oz,ox,om = vaxes.o
ny = 1; oy = 0.0; dy = 1
ivel = np.ascontiguousarray(ivel.reshape(vaxes.n,order='F').T).astype('float32')
vel = 1/smooth(1/ivel,rect1=40,rect2=30)

# Image perturbation
daxes,dimg = sep.read_file("wemvadimg.H")
dimg = dimg.reshape(daxes.n,order='F')[:,0,:].T
plot_img2d(dimg[140:652,:512].T,dz=dz,dx=dx,ox=ox,aspect=2.0,figname='./fig/annrev/wemvadimg.png')

# Slowness update
saxes,dbck = sep.read_file("wemvadslo.H")
dbck = np.ascontiguousarray(dbck.reshape(daxes.n,order='F')[:,0,:].T).astype('float32')
dbcksm = -smooth(dbck,rect1=4,rect2=4)
#plot_vel2d(dbcksm[140:652,:512].T,dz=dz,dx=dx,ox=ox,aspect=2.0)

# Anomaly
aaxes,iano = sep.read_wind("hale_tranos.H",nw=1,fw=0)
iano = np.ascontiguousarray(iano.reshape(aaxes.n,order='F').T).astype('float32')

# Create the slowness perturbation
dslo = -(vel - vel*iano)

plot_imgvelptb(syn[140:652,:512].T,1000*dslo[140:652,:512].T,dz=dz,dx=dx,ox=ox,thresh=5,
               velmin=-100,velmax=100,show=False,aagc=False,aspect=2.0,figname='./fig/annrev/velptb.png')

# Plot the  delta image
smin = np.min(syn); smax = np.max(syn)
fig = plt.figure(figsize=(10,5)); ax = fig.gca()
ax.imshow(syn[140:652,:512].T,extent=[ox,ox+nx*dx,oz+nz*dz,oz],cmap='gray',
          vmin=smin,vmax=smax,interpolation='bilinear',aspect=2.0)
im = ax.imshow(dbcksm[140:652,:512].T,extent=[ox,ox+nx*dx,oz+nz*dz,oz],cmap='seismic',
               interpolation='bilinear',vmax=500,vmin=-500,aspect=2.0,alpha=0.2)
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Z (km)',fontsize=15)
ax.tick_params(labelsize=15)
cbar_ax = fig.add_axes([0.81,0.15,0.02,0.70])
cbar = fig.colorbar(im,cbar_ax)
cbar.solids.set(alpha=1)
cbar.ax.tick_params(labelsize=15)
plt.savefig('./fig/annrev/wevmaadj.png',dpi=150,bbox_inches='tight',transparent=True)
#plt.show()

#plot_imgvelptb(syn[140:652,:512].T,dbcksm[140:652,:512].T,dz=dz,dx=dx,ox=ox,thresh=10,
#               velmin=-100,velmax=100,show=True,aagc=False,aspect=2.0)

#fig = plt.figure(figsize=(10,5)); ax = fig.gca()
#ax.imshow(stkw.T,extent=[30*dx+ox,ox+30*dx+nx*dx,nz*dz,0],interpolation='bilinear',
#          vmin=vmin*pclip,vmax=vmax*pclip,cmap='gray',aspect=3.0)
#ax.set_xlabel('X (km)',fontsize=15)
#ax.set_ylabel('Z (km)',fontsize=15)
#ax.tick_params(labelsize=15)
#plt.savefig("./fig/annrev/defocwhole.png",dpi=150,bbox_inches='tight',transparent=True)
#
#rect = patches.Rectangle((ox+30*dx,100*dz),nx*dx,256*dz,linewidth=2,edgecolor='yellow',facecolor='none')
#ax.add_patch(rect)
#plt.savefig("./fig/annrev/defocbox.png",dpi=150,bbox_inches='tight',transparent=True)
#
#plot_img2d(stkw[:,100:356].T,dx=dx,dz=dz,oz=100*dz,aspect=3.0,imin=vmin,imax=vmax,pclip=pclip,
#           figname='./fig/annrev/defoczoom.png')
#plot_img2d(foc.T,dx=dx,dz=dz,oz=100*dz,aspect=3.0,imin=vmin,imax=vmax,pclip=pclip,
#           figname='./fig/annrev/refoczoom.png')

