import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from resfoc.estro import refocusimg
import matplotlib.pyplot as plt
from utils.plot import plot_imgvelptb, plot_allanggats, plot_rhopicks, plot_anggatrhos
from utils.ptyprint import create_inttag

sep = seppy.sep()

raxes,ref = sep.read_file("./fant/reffant3.H")
ref = ref.reshape(raxes.n,order='F')

paxes,ptb = sep.read_file("./fant/ptbfant3.H")
ptb = ptb.reshape(paxes.n,order='F')

aaxes,ang = sep.read_file("./fant/angagcfant3.H")
ang = ang.reshape(aaxes.n,order='F')
ang = np.ascontiguousarray(ang.T).astype('float32')

[nz,na,nx,nro] = aaxes.n; [oz,oa,ox,oro] = aaxes.o
[dz,da,dx,dro] = aaxes.d

faaxes,fang = sep.read_file("./fant/imgafant3.H")
fang = fang.reshape(faaxes.n,order='F')
fang = np.ascontiguousarray(fang.T).astype('float32')

saxes,stk = sep.read_file("./fant/stkfant3.H")
stk = stk.reshape(saxes.n,order='F')
stk = np.ascontiguousarray(stk.T).astype('float32')

#plot_imgvelptb(ref,ptb,dz,dx,velmin=-100,velmax=100,thresh=5,agc=False,figname='./fig/fantrefptb.png',
#               hbar=0.5,barz=0.25)

vmin = -2.5; vmax = 2.5
#plot_allanggats(fang,dz/1000.0,dx/1000.0,jx=10,show=False,agc=True,vmin=vmin,vmax=vmax,
#    figname='./fig/fantfocangs.png',labelsize=15)
#
#plot_allanggats(ang[16],dz/1000.0,dx/1000.0,jx=10,show=False,agc=True,vmin=vmin,vmax=vmax,
#    figname='./fig/fantdefocangs.png',labelsize=15)
#
# Plot all gathers for each rho
#for iro in range(nro):
#  rho = oro + iro*dro
#  # Plot the angle gathers
#  plot_allanggats(ang[iro],dz/1000.0,dx/1000.0,jx=10,show=False,agc=False,vmin=vmin,vmax=vmax,
#      title=r'$\rho$=%.3f'%(rho),figname='./fig/fantdefocangs%d.png'%(iro),labelsize=15)
#  # Plot the stack
#  fig = plt.figure(figsize=(10,6))
#  ax = fig.gca()
#  ax.imshow(agc(stk[iro]).T,cmap='gray',interpolation='sinc',vmin=vmin,vmax=vmax,
#      extent=[0,nx*dx/1000.0,nz*dz/1000.0,0])
#  ax.set_xlabel('X (km)',fontsize=15)
#  ax.set_ylabel('Z (km)',fontsize=15)
#  ax.set_title(r'$\rho$=%.3f'%(rho),fontsize=15)
#  ax.tick_params(labelsize=15)
#  plt.savefig('./fig/fantdefocimg%d.png'%(iro),transparent=True,bbox_inches='tight',dpi=150)
#  plt.close()

# Agc the angle gathers and write them to file
#angagc = np.zeros(ang.shape,dtype='float32')
#for iro in range(nro):
#  print(iro)
#  angagc[iro] = agc(ang[iro])

#sep.write_file("./fant/angagcfant3.H",angagc.T,ds=[dz,da,dx,dro],os=[0,oa,0,oro])

# Read in agced angs
#aaxes,ang = sep.read_file("./fant/angagcfant3.H")
#ang = ang.reshape(aaxes.n,order='F')
#ang = np.ascontiguousarray(ang.T).astype('float32')

# Read in all semblance
saxes,smb = sep.read_file("./fant/smbfant3.H")
smb = smb.reshape(saxes.n,order='F')
smb = np.ascontiguousarray(smb.T).astype('float32')

# Read in all picks
paxes,pck = sep.read_file("./fant/rhofant3.H")
pck = pck.reshape(paxes.n,order='F')
pck = np.ascontiguousarray(pck.T).astype('float32')

# Plot every 20th image point
#for ix in range(0,nx,20):
#  tag = create_inttag(ix,1000)
#  plot_rhopicks(ang[:,ix,:,:],smb[ix,:,:],pck[ix],dro,dz/1000.0,oro,show=False,pclip=0.4,
#      figname='./fig/fant3smb-'+tag)
#  plot_anggatrhos(ang,ix,dz/1000.0,dx/1000.0,oro,dro,show=False,agc=True,pclip=0.4,
#      figname='./fig/fant3img-'+tag,fontsize=15,ticksize=15,wboxi=10,hboxi=6)

# Plot rho on top of image
#fsize = 15
#fig2 = plt.figure(figsize=(10,6))
#ax2 = fig2.gca()
#ax2.imshow(agc(stk[16,:,:]).T,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=vmin,vmax=vmax)
#im2 = ax2.imshow(pck.T,cmap='seismic',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear',vmin=0.98,vmax=1.02,alpha=0.1)
#ax2.set_xlabel('X (km)',fontsize=fsize)
#ax2.set_ylabel('Z (km)',fontsize=fsize)
#ax2.tick_params(labelsize=fsize)
#cbar_ax2 = fig2.add_axes([0.77,0.11,0.02,0.76])
#cbar2 = fig2.colorbar(im2,cbar_ax2,format='%.2f')
#cbar2.solids.set(alpha=1)
#cbar2.ax.tick_params(labelsize=fsize)
#cbar2.set_label(r'$\rho$',fontsize=fsize)
#plt.savefig('./fig/fantrho.png',bbox_inches='tight',transparent=True,dpi=150)
#plt.close()

# Plot image space rho on top of image and refocus the image
riaxes,rhoi = sep.read_file("./fant/rhoifant3.H")
rhoi = rhoi.reshape(riaxes.n,order='F')
rhoi = np.ascontiguousarray(rhoi.T).astype('float32')

rfii = refocusimg(stk,rhoi,dro)

sep.write_file("./fant/rfiifant3.H",rfii.T,ds=[dz,dx])

# Plot rho on top of image
fsize = 15
fig2 = plt.figure(figsize=(10,6))
ax2 = fig2.gca()
ax2.imshow(agc(stk[16,:,:]).T,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=vmin,vmax=vmax)
im2 = ax2.imshow(rhoi.T,cmap='seismic',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear',vmin=0.98,vmax=1.02,alpha=0.1)
ax2.set_xlabel('X (km)',fontsize=fsize)
ax2.set_ylabel('Z (km)',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
cbar_ax2 = fig2.add_axes([0.77,0.11,0.02,0.76])
cbar2 = fig2.colorbar(im2,cbar_ax2,format='%.2f')
cbar2.solids.set(alpha=1)
cbar2.ax.tick_params(labelsize=fsize)
cbar2.set_label(r'$\rho$',fontsize=fsize)
plt.savefig('./fig/fantrhoimg.png',bbox_inches='tight',transparent=True,dpi=150)
plt.close()

