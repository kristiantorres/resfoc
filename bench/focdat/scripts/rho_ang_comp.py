import numpy as np
import inpout.seppy as seppy
from resfoc.estro import estro_tgt,onehot2rho
from scaas.trismooth import smooth
from utils.movie import viewimgframeskey
from utils.plot import plot_anggatrhos
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in well-focused image
iaxes,img = sep.read_file("fltimg-00760.H")
#iaxes,img = sep.read_file("fltimg-01377.H")
img = img.reshape(iaxes.n,order='F')
izro = img[:,:,16]

# Read in velocity perturbation
paxes,ptb = sep.read_file("resfltptb-00760.H")
ptb = ptb.reshape(paxes.n,order='F')

# Read in residual migration image
raxes,res = sep.read_file("zoresfltimg-00760.H")
#raxes,res = sep.read_file("zoresfltimg-01377.H")
res = res.reshape(raxes.n,order='F')
[nz,nx,nro] = raxes.n; [dz,dx,dro] = raxes.d; [oz,ox,oro] = raxes.o

# Read in the angle gathers
aaxes,ang = sep.read_file("zangresfltimg-00760.H")
#aaxes,ang = sep.read_file("zangresfltimg-01377.H")
ang = ang.reshape(aaxes.n,order='F').T
[nz,na,nx,nro] = aaxes.n; [dz,da,dx,dro] = aaxes.d; [oz,oa,ox,oro] = aaxes.o

rho,lbls = estro_tgt(res.T,izro.T,dro,oro,strdx=64,strdz=64,onehot=True)

fsize = 14
fig1 = plt.figure(1,figsize=(14,7))
ax1 = fig1.gca()
im1 = ax1.imshow(smooth(rho.astype('float32'),rect1=100,rect2=100).T,extent=[0.0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0],
    cmap='seismic',vmax=1.025,vmin=0.975)
ax1.set_xlabel('X [km]',fontsize=fsize)
ax1.set_ylabel('Z [km]',fontsize=fsize)
ax1.tick_params(labelsize=fsize)
cbar_ax1 = fig1.add_axes([0.91,0.12,0.02,0.75])
cbar1 = fig1.colorbar(im1,cbar_ax1,format='%.2f')
cbar1.ax.tick_params(labelsize=fsize)
cbar1.set_label(r'$\rho$',fontsize=18)

fig3 = plt.figure(3,figsize=(14,7))
ax3 = fig3.gca()
im3 = ax3.imshow(rho.T,extent=[0.0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0],cmap='seismic',vmax=1.025,vmin=0.975)
ax3.set_xlabel('X [km]',fontsize=fsize)
ax3.set_ylabel('Z [km]',fontsize=fsize)
ax3.tick_params(labelsize=fsize)
cbar_ax3 = fig3.add_axes([0.91,0.12,0.02,0.75])
cbar3 = fig3.colorbar(im3,cbar_ax3,format='%.2f')
cbar3.ax.tick_params(labelsize=fsize)
cbar3.set_label(r'$\rho$',fontsize=18)

fig2 = plt.figure(2,figsize=(14,7))
ax2 = fig2.gca()
im2 = ax2.imshow(ptb,extent=[0.0,(nx)*dx/1000.0,(nz)*dz/1000.0,0.0],cmap='jet',vmin=-100,vmax=100)
ax2.set_xlabel('X [km]',fontsize=fsize)
ax2.set_ylabel('Z [km]',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
cbar_ax2 = fig2.add_axes([0.91,0.12,0.02,0.75])
cbar2 = fig2.colorbar(im2,cbar_ax2,format='%.2f')
cbar2.ax.tick_params(labelsize=fsize)
cbar2.set_label(r'm/s',fontsize=18)

#angs = ang[:,::10,:,:]
#nxs = angs.shape[1]
#angs = angs.reshape([nro,nxs*na,nz])
#viewimgframeskey(angs,pclip=0.5,ottl=oro,dttl=dro,ttlstring=r'$\rho$=%f')

#plot_anggatrhos(ang,600,dz/1000.0,dx/1000.0,oro,dro,show=False)
#plot_anggatrhos(ang,500,dz/1000.0,dx/1000.0,oro,dro,show=True)
plot_anggatrhos(ang,400,dz/1000.0,dx/1000.0,oro,dro,show=False,figname='./fig/anggatrho760',zmin=70,zmax=350,pclip=0.8)
#plot_anggatrhos(ang,300,dz/1000.0,dx/1000.0,oro,dro,show=False)
#plot_anggatrhos(ang,200,dz/1000.0,dx/1000.0,oro,dro,show=True)

#plt.show()
