import inpout.seppy as seppy
import numpy as np
from resfoc.semb import rho_semb, pick
from resfoc.estro import refocusimg, refocusang
import matplotlib.pyplot as plt
from genutils.plot import plot_rhopicks, plot_anggatrhos

sep = seppy.sep()

# Read in residually migrated gathers
saxes,storm = sep.read_file("resfaultfocust.H")
[nz,na,nx,nro] = saxes.n; [oz,oa,ox,oro] = saxes.o; [dz,da,dx,dro] = saxes.d
storm = storm.reshape(saxes.n,order='F').T

# Window the gathers
stormw = storm[:,:,20:,:]
sc1 = 0.2
smin = sc1*np.min(stormw); smax = sc1*np.max(stormw)

semb = rho_semb(stormw,gagc=True,rectz=15,nthreads=24)

rho = pick(semb,oro,dro,vel0=1.0,verb=True)

# Compute the stack
stkw = np.sum(stormw,axis=2)
sc2 = 0.2
kmin = sc2*np.min(stkw); kmax= sc2*np.max(stkw)

fsize = 16
#for ix in range(0,nx,50):
#  plot_anggatrhos(stormw[:,:,:,:850],ix,dz,dx,oro,dro,ox=ox,show=False,pclip=0.6,fontsize=fsize,ticksize=fsize,
#                  imgaspect=2.0,roaspect=0.01)
#  # Plot the picked
#  plot_rhopicks(stormw[:,ix,:,:850],semb[ix,:,:850],rho[ix,:850],dro,dz,oro,show=True,angaspect=0.01,
#                vmin=smin,vmax=smax,wspace=0.1,rhoaspect=0.04,pclip=1.1)

#for ix in range(nx):
#  plot_rhopicks(stormw[:,ix,:,:],semb[ix,:,:],rho[ix],dro,dz,oro,show=True,angaspect=0.01,
#                rhoaspect=0.05,vmin=smin,vmax=smax,wspace=0.1)

# Plot rho on top of stack
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(stkw[80,:850].T,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[ox,ox+nx*dx,oz+850*dz,oz],aspect='auto')
im = ax.imshow(rho[:,:850].T,cmap='seismic',interpolation='bilinear',vmin=0.95,vmax=1.05,
               extent=[ox,ox+nx*dx,oz+850*dz,oz],alpha=0.2,aspect='auto')
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
cbar_ax = fig.add_axes([0.925,0.212,0.02,0.7])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.solids.set(alpha=1)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r'$\rho$',fontsize=fsize)
#plt.savefig("./fig/rhoimg.png",dpi=150,bbox_inches='tight',transparent=True)
plt.show()

# Refocus the stack
rfi = refocusimg(stkw,rho,dro)
rfa = refocusang(stormw,rho,dro)

sep.write_file("faultfocussemb.H",semb.T,os=[oz,oro],ds=[dz,dro])
sep.write_file("faultfocusrho.H",rho.T,os=[oz,ox],ds=[dz,dx])
sep.write_file("faultfocusrfi.H",rfi.T,os=[oz,ox],ds=[dz,dx])
sep.write_file("faultfocusstk.H",stkw[80].T,os=[oz,ox],ds=[dz,dx])
sep.write_file("faultfocusrfa.H",rfa.T,os=[oz,0,ox],ds=[dz,da,dx])

