import inpout.seppy as seppy
import numpy as np
from genutils.plot import plot_rhopicks, plot_anggatrhos
from genutils.ptyprint import create_inttag

sep = seppy.sep()

# Read in all data
#aaxes,ang = sep.read_file('fltimgbigresangagcwrng.H')
aaxes,ang = sep.read_file('../focdat/dat/refocus/mltest/mltestdogang.H')
ang = ang.reshape(aaxes.n,order='F')
ang = np.ascontiguousarray(ang.T).astype('float32')

[nz,na,nx,nro] = aaxes.n; [oz,oa,ox,oro] = aaxes.o; [dz,da,dx,dro] = aaxes.d

# Read in all semblance
#saxes,smb = sep.read_file("rhosemblancenormwrng.H")
saxes,smb = sep.read_file("../focdat/dat/refocus/mltest/mltestdogsmb.H")
smb = smb.reshape(saxes.n,order='F')
smb = np.ascontiguousarray(smb.T).astype('float32')

# Read in all picks
paxes,pck = sep.read_file('../focdat/dat/refocus/mltest/mltestdogrho.H')
pck = pck.reshape(paxes.n,order='F')
pck = np.ascontiguousarray(pck.T).astype('float32')

# Plot every 10th image point
for ix in range(100,nx,20):
  tag = create_inttag(ix,1000)
  plot_rhopicks(ang[:,ix,:,:],smb[ix,:,:],pck[ix],dro,dz/1000.0,oro,show=True,pclip=0.8)
  plot_anggatrhos(ang,ix,dz/1000.0,dx/1000.0,oro,dro,show=True,pclip=0.4,fontsize=15,ticksize=15,wboxi=10,hboxi=6)

#  plot_rhopicks(ang[:,ix,:,:],smb[ix,:,:],pck[ix],dro,dz/1000.0,oro,show=False,pclip=0.8,
#      figname='./fig/rhopicks/smb-'+tag)
#  plot_anggatrhos(ang,ix,dz/1000.0,dx/1000.0,oro,dro,show=False,pclip=0.4,
#      figname='./fig/rhopicks/img-'+tag,fontsize=15,ticksize=15,wboxi=10,hboxi=6)

