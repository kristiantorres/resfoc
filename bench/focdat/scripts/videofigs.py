import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from genutils.ptyprint import create_inttag
import matplotlib.pyplot as plt
from genutils.plot import plot_imgvelptb, plot_cubeiso, plot_allanggats, plot_rhopicks, plot_anggatrhos

sep = seppy.sep()

# Reflectivity
raxes,ref = sep.read_file("./dat/focdefoc/mltestref.H")
ref = ref.reshape(raxes.n,order='F')

[nz,nx] = raxes.n; [dz,dx] = raxes.d

# Velocity perturbation
paxes,ptb = sep.read_file("./dat/focdefoc/mltestptb.H")
ptb = ptb.reshape(paxes.n,order='F')

# Well-focused image
faxes,fog = sep.read_file("./dat/focdefoc/mltestfog.H")
fog = np.ascontiguousarray(fog.reshape(faxes.n,order='F').T).astype('float32')
zofog = fog[16]
fogg = agc(zofog).T

nx = 400; fx = 305
nz = 300; fz = 120

foggw = fogg[fz:fz+nz,fx:fx+nx]

# Defocused image
daxes,dog = sep.read_file("./dat/focdefoc/mltestdog.H")
dog = np.ascontiguousarray(dog.reshape(daxes.n,order='F').T).astype('float32')
zodog = dog[16]
dogg = agc(zodog).T

zodogw = zodog[fx:fx+nx,fz:fz+nz].T
doggw = dogg[fz:fz+nz,fx:fx+nx]

# Residual migration stacks
saxes,stk = sep.read_file("./dat/refocus/mltest/mltestdogstk2.H")
stk = np.ascontiguousarray(stk.reshape(saxes.n,order='F').T)

nro = saxes.n[2]; oro = saxes.o[2]; dro = saxes.d[2]

# Read short angle gathers
aaxes,ang = sep.read_file("./dat/focdefoc/mltestfag.H")
ang = ang.reshape(aaxes.n,order='F').T
angw = ang[fx:fx+nx,:,fz:fz+nz]
#angww = angw[74:158,:,44:128]

# Windowing parameters
fxl =  49; nxl = 400
fzl = 120; nzl = 300

# Read residually migrated angle gathers
laxes,lng = sep.read_file("./dat/refocus/mltest/mltestdogang.H")
lng = lng.reshape(laxes.n,order='F').T
lngw = lng[:,fxl:fxl+nxl,:,fzl:fzl+nzl]

#laxes,lng= sep.read_file("./dat/refocus/mltest/mltestdogang2mask.H")
#lng = lng.reshape(laxes.n,order='F').T
#lngw = lng[:,fxl:fxl+nxl,:,fzl:fzl+nzl]

# Read in all semblance
saxes,smb = sep.read_file("../focdat/dat/refocus/mltest/mltestdogsmb.H")
smb = smb.reshape(saxes.n,order='F')
smb = np.ascontiguousarray(smb.T).astype('float32')
smbw = smb[fxl:fxl+nxl,:,fzl:fzl+nzl]

#saxes,smb = sep.read_file("../focdat/dat/refocus/mltest/mltestdogsmbmask2.H")
#smb = smb.reshape(saxes.n,order='F')
#smb = np.ascontiguousarray(smb.T).astype('float32')
#smbw = smb[fxl:fxl+nxl,:,fzl:fzl+nzl]

# Read in all picks
paxes,pck = sep.read_file('../focdat/dat/refocus/mltest/mltestdogrho.H')
pck = pck.reshape(paxes.n,order='F')
pck = np.ascontiguousarray(pck.T).astype('float32')
pckw = pck[fxl:fxl+nxl,fzl:fzl+nzl]

#paxes,pck = sep.read_file('../focdat/dat/refocus/mltest/mltestdogrhomask2.H')
#pck = pck.reshape(paxes.n,order='F')
#pck = np.ascontiguousarray(pck.T).astype('float32')
#pckw = pck[fxl:fxl+nxl,fzl:fzl+nzl]

caxes,cnn = sep.read_file('../ml/rhoangcnn3.H')
cnn = cnn.reshape(caxes.n,order='F')
cnn = np.ascontiguousarray(cnn.T).astype('float32')
cnnw = cnn[fxl:fxl+nxl,fzl:fzl+nzl]

# Read in refocused image
#iaxes,rfi = sep.read_file('../focdat/dat/refocus/mltest/mltestdogrfi.H')
#rfi = rfi.reshape(iaxes.n,order='F')
#rfi = np.ascontiguousarray(rfi.T).astype('float32')
#rfig = agc(rfi)
#rfiw  = rfi [fxl:fxl+nxl,fzl:fzl+nzl]
#rfigw = rfig[fxl:fxl+nxl,fzl:fzl+nzl]

iaxes,rfi = sep.read_file('../focdat/dat/refocus/mltest/mltestdogrfimask2.H')
rfi = rfi.reshape(iaxes.n,order='F')
rfi = np.ascontiguousarray(rfi.T).astype('float32')
rfig = agc(rfi)
rfiw  = rfi [fxl:fxl+nxl,fzl:fzl+nzl]
rfigw = rfig[fxl:fxl+nxl,fzl:fzl+nzl]

#vmin=-2.5; vmax=2.5
#plot_allanggats(angw,dz/1000.0,dx/1000.0,jx=10,show=False,agc=True,vmin=vmin,vmax=vmax,
#    figname='./fig/videofigs/focusedangs.png',labelsize=16,wbox=8,hbox=8,xmin=fx*dx/1000.0,xmax=(fx+nx)*dx/1000.0,
#    zmin=fz*dz/1000.0,zmax=(fz+nz)*dz/1000.0)

#acub = np.transpose(angww,(1,2,0))
#acub = np.transpose(angw,(1,2,0))

#xmin = (fx+84)*0.01; zmin = (fz+54)*0.01
#xmin = (fx)*0.01; zmin = (fz)*0.01

#plot_cubeiso(acub,os=[-70.0,zmin,xmin],ds=[2.22,0.01,0.01],stack=True,show=False,hbox=8,wbox=8,elev=15,
#                 x1label='\nX (km)',x2label='\nAngle'+r'($\degree$)',x3label='\nZ (km)',verb=False,figname='./fig/videofigs/ancubefoc.png')
#plot_cubeiso(acub,os=[-70.0,zmin,xmin],ds=[2.22,0.01,0.01],stack=True,show=True,hbox=8,wbox=8,elev=15,
#                 x1label='\nX (km)',x2label='\nAngle'+r'($\degree$)',x3label='\nZ (km)',verb=False)

#plot_imgvelptb(ref,ptb,dz,dx,velmin=-100,velmax=100,thresh=5,agc=False,figname='./fig/videofigs/mltestrefptb.pdf',
#               hbar=0.5,barz=0.25,alpha=0.5)

dx /= 1000.0; dz /= 1000.0
xmin = fx*dx; xmax = (fx+nx)*dx
zmin = fz*dz; zmax = (fz+nz)*dz
#
## Plot the well-focused image
fsize = 16
#fig = plt.figure(figsize=(10,10)); ax = fig.gca()
#ax.imshow(foggw,cmap='gray',interpolation='sinc',extent=[xmin,xmax,zmax,zmin],vmin=-2.5,vmax=2.5)
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/videofigs/mltestfog.png',bbox_inches='tight',dpi=150,transparent=True)
#plt.close()
#
#fig = plt.figure(figsize=(8,8)); ax = fig.gca()
#ax.imshow(doggw,cmap='gray',interpolation='sinc',extent=[xmin,xmax,zmax,zmin],vmin=-2.5,vmax=2.5)
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/videofigs/mltestdog.pdf',bbox_inches='tight',dpi=150,transparent=True)
#plt.close()

# Windowing parameters
fx2 =  49; nx2 = 400
fz2 = 120; nz2 = 300

# Plot all gathers for each rho
vmin = -2.5; vmax=2.5
for iro in range(nro):
#  rho = oro + iro*dro
#  # Plot the angle gathers
#  plot_allanggats(lngw[iro],dz,dx,jx=10,show=False,agc=True,vmin=vmin,vmax=vmax,
#                  title=r'$\rho$=%.5f'%(rho),figname='./fig/videofigs/angro%d.png'%(iro),labelsize=16,xmin=fx*dx,xmax=(fx+nx)*dx,
#                  zmin=(fz)*dz,zmax=(fz+nz)*dz)
  plot_allanggats(lngw[iro],dz,dx,jx=10,show=False,agc=True,vmin=vmin,vmax=vmax,
                  figname='./fig/videofigs/angro%d.png'%(iro),labelsize=16,xmin=fx*dx,xmax=(fx+nx)*dx,
                  zmin=(fz)*dz,zmax=(fz+nz)*dz)
#  # Plot the stack
#  fig = plt.figure(figsize=(8,8))
#  ax = fig.gca()
#  ax.imshow(agc(stk[iro,fx2:fx2+nx2,fz2:fz2+nz2]).T,cmap='gray',interpolation='sinc',vmin=vmin,vmax=vmax,
#      extent=[fx*dx,(fx+nx)*dx,(fz+nz)*dz,fz*dz])
#  ax.set_xlabel('X (km)',fontsize=fsize)
#  ax.set_ylabel('Z (km)',fontsize=fsize)
#  #ax.set_title(r'$\rho$=%.5f'%(rho),fontsize=fsize)
#  ax.tick_params(labelsize=fsize)
#  plt.savefig('./fig/videofigs/stkrho%d.png'%(iro),transparent=True,bbox_inches='tight',dpi=150)
#  plt.close()

# Defocused angle gathers
#plot_allanggats(lngw[20],dz,dx,jx=10,show=False,agc=True,vmin=vmin,vmax=vmax,
#                figname='./fig/videofigs/defocusedangs.png',labelsize=16,xmin=fx*dx,xmax=(fx+nx)*dx,
#                zmin=(fz)*dz,zmax=(fz+nz)*dz)

#plt.show()

# Plot every 10th image point
#xs = [40,100,210,270,310,360]
#xs = [300]
#for ix in xs:
##for ix in range(0,nxl,10):
#  tag = create_inttag(ix,10)
#  plot_rhopicks(lngw[:,ix,:,:],smbw[ix,:,:],pckw[ix],dro,dz,oro,show=True,doagc=False,vmin=-0.0025,vmax=0.0025,
#                wspace=0.15,zmin=fz*dz,zmax=(fz+nz)*dz,angaspect=0.01,rhoaspect=0.02,cnnpck=cnnw[ix],
#                figname='./fig/videofigs/smbpickscnnfull/smbpicks%s'%(tag))
#  plot_anggatrhos(lngw,ix,dz,dx,oro,dro,ox=fx,show=False,pclip=0.4,fontsize=fsize,ticksize=fsize,wboxi=10,hboxi=6,
#                  zmin=fz,zmax=fz+nz,xmin=fx,xmax=fx+nx,figname='./fig/videofigs/smbpickscnnfull/rhoang%s'%(tag))

# Plot rho on the image
#fig2 = plt.figure(figsize=(10,10))
#ax2 = fig2.gca()
####ax2.imshow(zodogw,cmap='gray',extent=[fx*dx,(fx+nx)*dx,(fz+nz)*dz,fz*dz],interpolation='sinc',vmin=np.min(zodogw)*0.4,vmax=np.max(zodogw)*0.4)
#ax2.imshow(doggw,cmap='gray',extent=[fx*dx,(fx+nx)*dx,(fz+nz)*dz,fz*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
#im2 = ax2.imshow(pckw.T,cmap='seismic',extent=[fx*dx,(fx+nx)*dx,(fz+nz)*dz,fz*dz],interpolation='bilinear',vmin=0.975,vmax=1.025,alpha=0.2)
#ax2.set_xlabel('X (km)',fontsize=fsize)
#ax2.set_ylabel('Z (km)',fontsize=fsize)
#ax2.tick_params(labelsize=fsize)
#cbar_ax2 = fig2.add_axes([0.925,0.205,0.02,0.58])
#cbar2 = fig2.colorbar(im2,cbar_ax2,format='%.2f')
#cbar2.solids.set(alpha=1)
#cbar2.ax.tick_params(labelsize=fsize)
#cbar2.set_label(r'$\rho$',fontsize=fsize)
##plt.show()
#plt.savefig('./fig/videofigs/rhosmbmask2.png',bbox_inches='tight',transparent=True,dpi=150)
###plt.close()

# Plot the refocused image
#fig2 = plt.figure(figsize=(10,10))
#ax2 = fig2.gca()
#ax2.imshow(rfiw.T,cmap='gray',extent=[fx*dx,(fx+nx)*dx,(fz+nz)*dz,fz*dz],interpolation='sinc',vmin=np.min(rfiw)*0.4,vmax=np.max(rfiw)*0.4)
#ax2.imshow(rfigw.T,cmap='gray',extent=[fx*dx,(fx+nx)*dx,(fz+nz)*dz,fz*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
#ax2.set_xlabel('X (km)',fontsize=fsize)
#ax2.set_ylabel('Z (km)',fontsize=fsize)
#ax2.tick_params(labelsize=fsize)
##plt.show()
#plt.savefig('./fig/videofigs/rfismbmask2.png',bbox_inches='tight',transparent=True,dpi=150)
#plt.close()

# Plot single angle angle gather
#fig = plt.figure(figsize=(10,10)); ax = fig.gca()
#ax.imshow(lngw[20,310,:,:].T,cmap='gray',interpolation='sinc',extent=[-70,70,(fz+nz)*dz,fz*dz],vmin=-0.0025,vmax=0.0025,aspect=200)
#ax.set_xlabel(r'Angle ($\degree$)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/videofigs/anggat.png',bbox_inches='tight',transparent=True,dpi=150)
#plt.show()
