import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from resfoc.estro import estro_varimax, refocusimg
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.focuslabels import varimax
from genutils.movie import viewimgframeskey
import matplotlib.pyplot as plt

sep = seppy.sep()

faxes,fog = sep.read_file("dat/refocus/mltest/mltestfogstk.H")
fog = np.ascontiguousarray(fog.reshape(faxes.n,order='F').T)

[nz,nx,nro] = faxes.n; [oz,ox,oro] = faxes.o; [dz,dx,dro] = faxes.d

daxes,dog = sep.read_file("dat/refocus/mltest/mltestdogstk.H")
dog = np.ascontiguousarray(dog.reshape(daxes.n,order='F').T)

fogg = agc(fog); dogg = agc(dog)

doggt = np.transpose(dogg,(0,2,1))
rho,entropy = estro_varimax(doggt,dro,oro)

# Plotting window
fx =  49; nx = 400
fz = 120; nz = 300
dx /= 1000.0; dz /= 1000.0

ro1 = doggt[21,fz:fz+nz,fx:fx+nx]
rhow = rho[fz:fz+nz,fx:fx+nx]

fsize = 16
fig3 = plt.figure(2,figsize=(10,10)); ax3 = fig3.gca()
ax3.imshow(ro1,cmap='gray',extent=[fx*dx,(fx+nx)*dx,(fz+nz)*dz,fz*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
im3 = ax3.imshow(rhow,cmap='seismic',extent=[fx*dx,(fx+nx)*dx,(fz+nz)*dz,fz*dz],interpolation='bilinear',vmin=0.975,vmax=1.025,alpha=0.2)
ax3.set_xlabel('X (km)',fontsize=fsize)
ax3.set_ylabel('Z (km)',fontsize=fsize)
ax3.tick_params(labelsize=fsize)
ax3.set_title(r"Entropy focusing",fontsize=fsize)
cbar_ax3 = fig3.add_axes([0.925,0.205,0.02,0.58])
cbar3 = fig3.colorbar(im3,cbar_ax3,format='%.2f')
cbar3.solids.set(alpha=1)
cbar3.ax.tick_params(labelsize=fsize)
cbar3.set_label(r'$\rho$',fontsize=fsize)
plt.savefig('./fig/rhovarimax.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

rfi  = refocusimg(doggt,rho,dro)

rfiw = rfi[fz:fz+nz,fx:fx+nx]

fig4 = plt.figure(3,figsize=(10,10)); ax4 = fig4.gca()
ax4.imshow(rfiw,cmap='gray',extent=[fx*dx,(fx+nx)*dx,(fz+nz)*dz,fz*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax4.set_xlabel('X (km)',fontsize=fsize)
ax4.set_ylabel('Z (km)',fontsize=fsize)
ax4.set_title('Entropy',fontsize=fsize)
ax4.tick_params(labelsize=fsize)
plt.savefig('./fig/rfientropy.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

## Create patch extractor
#nzp   = 64;    nxp   = 64
#strdz = int(nzp/2); strdx = int(nxp/2)
#pe = PatchExtractor((nzp,nxp),stride=(strdx,strdz))
#fptch = pe.extract(fogg[20])
#
#numpx = fptch.shape[0]; numpz = fptch.shape[1]
#
#movie = np.zeros([nro,nzp,nxp])
#
#norm = np.zeros(nro)
#for iro in range(nro):
#  ptch = pe.extract(dogg[iro])
#  movie[iro] = ptch.reshape([numpx*numpz,nxp,nzp])[65]
#  norm[iro] = varimax(movie[iro])
#  #print("rho=%.4f,varimax=%.2f"%(iro*dro + oro,norm[iro]))
#
#fsize = 16
#rho = np.linspace(oro,oro+(nro-1)*dro,nro)
#viewimgframeskey(movie,pclip=0.5,interp='sinc',ottl=oro,dttl=dro,ttlstring=r'$\rho$=%.4f',show=False)
#fig = plt.figure(); ax = fig.gca()
#ax.plot(rho,norm)
#ax.set_xlabel(r'$\rho$',fontsize=fsize)
#ax.set_ylabel('varimax',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.show()

