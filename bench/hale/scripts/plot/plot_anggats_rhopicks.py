import inpout.seppy as seppy
import numpy as np
from genutils.plot import plot_rhopicks, plot_anggatrhos
from genutils.ptyprint import create_inttag

sep = seppy.sep()

# Read in residually migrated image
saxes,storm = sep.read_file("resfaultfocuswindt.H")
#saxes,storm = sep.read_file("resfaultfocuswindtmute.H")
nz,na,nx,nro = saxes.n; oz,oa,ox,oro = saxes.o; dz,da,dx,dro = saxes.d
storm = storm.reshape(saxes.n,order='F').T

# Window the gathers
stormw = storm[:,:,32:,100:356]
sc1 = 0.2
smin = sc1*np.min(stormw); smax = sc1*np.max(stormw)

# Read in semblance
baxes,semb = sep.read_file("faultfocussembwind.H")
semb = semb.reshape(baxes.n,order='F').T

# Read in Rho
raxes,rho = sep.read_file("faultfocusrhowind.H")
rho = rho.reshape(raxes.n,order='F').T

# Compute the stack
stkw = np.sum(stormw,axis=2)
sc2 = 0.2
kmin = sc2*np.min(stkw); kmax= sc2*np.max(stkw)

fsize = 16
for ix in range(0,nx,50):
  tag = create_inttag(ix,nx)
  plot_anggatrhos(stormw[48:112],ix,dz,dx,oro=0.96,dro=dro,ox=ox,show=False,pclip=0.6,fontsize=fsize,ticksize=fsize,
                  imgaspect=2.0,roaspect=0.02,figname='./fig/halepicks/anggatrhos-%s'%(tag))
  # Plot the picked
  plot_rhopicks(stormw[48:112,ix,:,:],semb[ix,48:112,:],rho[ix,:],dro,dz,oro=0.96,show=False,angaspect=0.04,
                vmin=smin,vmax=smax,wspace=0.1,rhoaspect=0.16,pclip=1.1,figname='./fig/halepicks/rhopick-%s'%(tag))

