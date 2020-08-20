"""
Makes plots from prestack residual migration

@author: Joseph Jennings
@version: 2020.04.05
"""
import numpy as np
import inpout.seppy as seppy
from resfoc.gain import agc
import matplotlib.pyplot as plt
from genutils.movie import makemovie_mpl, viewimgframeskey, makemovietb_mpl, makemoviesbs_mpl
from genutils.plot import plot_anggatrhos

sep = seppy.sep()

# Read in stack
saxes,stk = sep.read_file("resangwrngconvstk.H")
[nz,nx,nro] = saxes.n; [oz,ox,oro] = saxes.o; [dz,dx,dro] = saxes.d
stk = stk.reshape(saxes.n,order='F')
stk = np.ascontiguousarray(stk.T).astype('float32')

gne = agc(stk)

#viewimgframeskey(gne[:,100:610,:],transp=True,pclip=0.5)
#
#bxidx = 20; exidx = 1000
#makemovie_mpl(gne.T[:,bxidx:exidx,:],'./fig/stkrmig',qc=False,pclip=0.5,ttlstring=r'$\rho$=%.4f',ottl=oro,dttl=dro,
#    labelsize=15,hbox=6,wbox=10,ticksize=15,xlabel='X (km)',ylabel='Z (km)',
#    zmax=(nz-1)*dz/1000.0,xmin=bxidx*dx/1000.0,xmax=(exidx-1)*dx/1000.0)

# Read in angle depth gathers
aaxes,ang = sep.read_file("resangwrngconv.H")
[nz,na,nx,nro] = aaxes.n; [oz,oa,ox,oro] = aaxes.o; [dz,da,dx,dro] = aaxes.d
ang = ang.reshape(aaxes.n,order='F')
ang = np.ascontiguousarray(ang.T).astype('float32')

#jx = 10
#angss = ang[:,::jx,:,:]
#nxs = angss.shape[1]
#angssr = angss.reshape([nro,na*nxs,nz])
#
#anggne = agc(angssr)
#
##viewimgframeskey(anggne,transp=True,pclip=0.5)
#
#ntot = anggne.shape[1]
#minidx = 10*na; maxidx = int(ntot) - 10*na
#ixmin = int(minidx/na)*jx; ixmax = int(maxidx/na)*jx

#makemovie_mpl(anggne[:,minidx:maxidx,75:450].T,'./fig/angrmig',qc=True,pclip=0.5,ttlstring=r'$\rho$=%.4f',ottl=oro,dttl=dro,
#    labelsize=15,hbox=6,wbox=10,ticksize=15,xlabel='X (km)',ylabel='Z (km)',zmin=(75)*dz/1000.0,
#    zmax=(450-1)*dz/1000.0,xmin=ixmin*dx/1000.0,xmax=ixmax*dx/1000.0)

#minidx = 10*na; maxidx = int(ntot/2) + 10*na
#minidx = int(ntot/2); maxidx = int(ntot) - 10*na
#ixmin = int(minidx/na)*jx; ixmax = int(maxidx/na)*jx
##izmin=75; izmax=400
#izmin=75; izmax=450
#makemovietb_mpl(anggne[:,minidx:maxidx,izmin:izmax].T,gne[:,ixmin:ixmax,izmin:izmax].T,'./fig/angstkrmigright',qc=True,pclip=0.5,
#    ttlstring1=r'$\rho$=%.4f',ottl1=oro,dttl1=dro,
#    labelsize=15,hbox=7,wbox=15,ticksize=15,xlabel2='X (km)',ylabel1='Z (km)',ylabel2='Z (km)',
#    xmin1=ixmin*dx/1000.0,xmax1=ixmax*dx/1000.0,xmin2=ixmin*dx/1000.0,xmax2=ixmax*dx/1000.0,
#    zmin1=(izmin)*dz/1000.0,zmax1=(izmax)*dz/1000.0,zmin2=(izmin)*dz/1000.0,zmax2=(izmax)*dz/1000.0,aspect1=0.5,aspect2=1.0)

#makemoviesbs_mpl(anggne[:,minidx:maxidx,50:400].T,gne[:,ixmin:ixmax,50:400].T,'./fig/angstkrmig',qc=True,pclip=0.5,
#    ttlstring1=r'$\rho$=%.4f',ottl1=oro,dttl1=dro,
#    labelsize=15,hbox=7,wbox=15,ticksize=15,xlabel2='X (km)',ylabel1='Z (km)',ylabel2='Z (km)',
#    xmin1=ixmin*dx/1000.0,xmax1=ixmax*dx/1000.0,xmin2=ixmin*dx/1000.0,xmax2=ixmax*dx/1000.0,
#    zmin1=(50)*dz/1000.0,zmax1=(400)*dz/1000.0,zmin2=(50)*dz/1000.0,zmax2=(400)*dz/1000.0,aspect1=0.5,aspect2=1.0)


# Plot single angle gathers
show=False
plot_anggatrhos(ang,xloc=180,dz=dz/1000.0,dx=dx/1000.0,oro=oro,dro=dro,figname='./fig/anggat180',agc=True,
    wboxi=14,hboxi=6,xmin=20,xmax=999,zmin=50,zmax=400,labelsize=15,show=show)
plot_anggatrhos(ang,xloc=350,dz=dz/1000.0,dx=dx/1000.0,oro=oro,dro=dro,figname='./fig/anggat350',agc=True,
    wboxi=14,hboxi=6,xmin=20,xmax=999,zmin=50,zmax=400,labelsize=15,show=show)
plot_anggatrhos(ang,xloc=500,dz=dz/1000.0,dx=dx/1000.0,oro=oro,dro=dro,figname='./fig/anggat500',agc=True,
    wboxi=14,hboxi=6,xmin=20,xmax=999,zmin=50,zmax=400,labelsize=15,show=show)
plot_anggatrhos(ang,xloc=700,dz=dz/1000.0,dx=dx/1000.0,oro=oro,dro=dro,figname='./fig/anggat700',agc=True,
    wboxi=14,hboxi=6,xmin=20,xmax=999,zmin=50,zmax=400,labelsize=15,show=show)
plot_anggatrhos(ang,xloc=830,dz=dz/1000.0,dx=dx/1000.0,oro=oro,dro=dro,figname='./fig/anggat830',agc=True,
    wboxi=14,hboxi=6,xmin=20,xmax=999,zmin=50,zmax=400,labelsize=15,show=show)

