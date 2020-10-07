import inpout.seppy as seppy
import numpy as np
from resfoc.semb import rho_semb, pick
from resfoc.estro import refocusimg
from scaas.velocity import salt_mask
import matplotlib.pyplot as plt
from genutils.plot import plot_rhopicks, plot_anggatrhos

sep = seppy.sep()

# Read in residually migrated gathers
saxes,storm = sep.read_file("resmskoverwt.H")
#saxes,storm = sep.read_file("sigsbeewrngposrest.H")
[nz,na,nx,nro] = saxes.n; [oz,oa,ox,oro] = saxes.o; [dz,da,dx,dro] = saxes.d
storm = storm.reshape(saxes.n,order='F').T

# Read in the velocity model to mask the salt
vaxes,vel = sep.read_file("sigoverw_velint.H")
vel = vel.reshape(vaxes.n,order='F').T
velw = vel[20:480,:,240:1150]

# Window the gathers
stormw = storm[:,20:480,31:,240:1150]
sc1 = 0.1
smin = sc1*np.min(stormw); smax = sc1*np.max(stormw)

semb = rho_semb(stormw,gagc=True,rectz=15,nthreads=24)

rectx = 20
rho = pick(semb,oro,dro,vel0=1.0,verb=True,rectx=rectx)

# Compute the stack
stkw = np.sum(stormw,axis=2)
sc2 = 0.2
kmin = sc2*np.min(stkw); kmax= sc2*np.max(stkw)

fsize = 16
#for ix in range(0,nx,50):
#  plot_anggatrhos(stormw,ix,dz,dx,oro,dro,ox=ox,show=False,pclip=0.4,fontsize=fsize,ticksize=fsize,
#                  imgaspect=2.0,roaspect=0.005,figname='./fig/ressemb/imgline%d'%(ix))
#  # Plot the picked
#  plot_rhopicks(stormw[:,ix,:,:],semb[ix,:,:],rho[ix],dro,dz,oro,show=True,angaspect=0.005,
#                vmin=smin,vmax=smax,wspace=0.0,figname='./fig/ressemb/rhopick%d'%(ix))

# Mask the rho image
msk,rhomsk = salt_mask(rho,velw,saltvel=4.5)
idx = rhomsk == 0.0
rhomsk[idx] = 1.0

# Plot rho on top of stack
fsize = 15
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(stkw[20].T,interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,
          extent=[ox+20*dx,nx*dx,1150*dz,240*dz+oz],aspect=2.0)
im = ax.imshow(rhomsk.T,cmap='seismic',interpolation='bilinear',vmin=0.975,vmax=1.025,
               extent=[ox+20*dx,nx*dx,1150*dz,240*dz+oz],alpha=0.2,aspect=2.0)
cbar_ax = fig.add_axes([0.925,0.212,0.02,0.58])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.solids.set(alpha=1)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r'$\rho$',fontsize=fsize)
#plt.savefig("./fig/rhoimgpos.png",dpi=150,bbox_inches='tight',transparent=True)
plt.show()

# Refocus the stack
rfi = refocusimg(stkw,rhomsk,dro)

sep.write_file("sigsemb%d.H"%(rectx),semb.T,os=[oz,oro],ds=[dz,dro])
sep.write_file("sigrho%d.H"%(rectx),rho.T,os=[oz,ox+20*dx],ds=[dz,dx])
sep.write_file("sigrfi%d.H"%(rectx),rfi.T,os=[oz,ox+20*dx],ds=[dz,dx])
sep.write_file("stkwind%d.H"%(rectx),stkw[20].T,os=[oz,ox+20*dx],ds=[dz,dx])

