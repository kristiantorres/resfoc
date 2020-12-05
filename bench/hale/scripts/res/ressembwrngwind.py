import inpout.seppy as seppy
import numpy as np
from resfoc.semb import rho_semb, pick
from resfoc.estro import refocusimg, refocusang
from genutils.plot import plot_rhopicks, plot_anggatrhos, plot_rhoimg2d
from genutils.ptyprint import create_inttag

sep = seppy.sep()

# Read in residually migrated gathers
saxes,storm = sep.read_file("resfaultfocuswindt.H")
#saxes,storm = sep.read_file("resfaultfocuswindtmute.H")
nz,na,nx,nro = saxes.n; oz,oa,ox,oro = saxes.o; dz,da,dx,dro = saxes.d
storm = storm.reshape(saxes.n,order='F').T

# Window the gathers
stormw = storm[:,:,32:,:]
sc1 = 0.2
smin = sc1*np.min(stormw); smax = sc1*np.max(stormw)

semb = rho_semb(stormw,gagc=True,rectz=15,nthreads=24)

rho = pick(semb,oro,dro,vel0=1.0,verb=True)

# Compute the stack
stkw = np.sum(stormw,axis=2)
sc2 = 0.2
kmin = sc2*np.min(stkw); kmax= sc2*np.max(stkw)

fsize = 16
for ix in range(0,nx,50):
  tag = create_inttag(ix,nx)
  plot_anggatrhos(stormw[48:112],ix,dz,dx,oro=0.96,dro=dro,ox=ox,show=False,pclip=0.6,fontsize=fsize,ticksize=fsize,
                  imgaspect=2.0,roaspect=0.02,figname='./fig/halepicks2/anggatrhos-%s'%(tag))
  # Plot the picked
  plot_rhopicks(stormw[48:112,ix,:,:],semb[ix,48:112,:],rho[ix,:],dro,dz,oro=0.96,show=False,angaspect=0.02,
                vmin=smin,vmax=smax,wspace=0.1,rhoaspect=0.08,pclip=1.1,figname='./fig/halepicks2/muterhopick-%s'%(tag))

# Refocus the stack
rfi = refocusimg(stkw,rho,dro)
rfa = refocusang(stormw,rho,dro)

sembw = semb[:,:,100:356]
stkww = stkw[80,:,100:356]
rhow  = rho[:,100:356]
rfiw  = rfi[:,100:356]
rfaw  = rfa[:,:,100:356]

# Window to the target region
plot_rhoimg2d(stkww.T,rhow.T,dx=dx,dz=dz,ox=ox,oz=100*dz,aspect=2.0)

#sep.write_file("faultfocussembmutwind.H",sembw.T,os=[oz,oro],ds=[dz,dro])
#sep.write_file("faultfocusrhomutwind.H",rhow.T,os=[oz,ox],ds=[dz,dx])
#sep.write_file("faultfocusrfimutwind.H",rfiw.T,os=[oz,ox],ds=[dz,dx])
#sep.write_file("faultfocusstkmutwind.H",stkww.T,os=[oz,ox],ds=[dz,dx])
#sep.write_file("faultfocusrfamutwind.H",rfaw.T,os=[oz,0,ox],ds=[dz,da,dx])

#sep.write_file("faultfocussembmutwindfail.H",sembw.T,os=[oz,oro],ds=[dz,dro])
#sep.write_file("faultfocusrhomutwindfail.H",rhow.T,os=[oz,ox],ds=[dz,dx])
#sep.write_file("faultfocusrfimutwindfail.H",rfiw.T,os=[oz,ox],ds=[dz,dx])
#sep.write_file("faultfocusstkmutwindfail.H",stkww.T,os=[oz,ox],ds=[dz,dx])
#sep.write_file("faultfocusrfamutwindfail.H",rfaw.T,os=[oz,0,ox],ds=[dz,da,dx])

