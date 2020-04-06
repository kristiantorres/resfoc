"""
Makes plots from prestack residual migration

@author: Joseph Jennings
@version: 2020.04.05
"""
import numpy as np
import inpout.seppy as seppy
from resfoc.gain import agc
import matplotlib.pyplot as plt
from utils.movie import makemovie_mpl, viewimgframeskey

sep = seppy.sep()

# Read in stack
#saxes,stk = sep.read_file("resangwrngconvstk.H")
#[nz,nx,nro] = saxes.n; [oz,ox,oro] = saxes.o; [dz,dx,dro] = saxes.d
#stk = stk.reshape(saxes.n,order='F')
#stk = np.ascontiguousarray(stk.T).astype('float32')
#
#gne = agc(stk)
#
##viewimgframeskey(gne,transp=True,pclip=0.5)
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

jx = 10
angss = ang[:,::jx,:,:]
nxs = angss.shape[1]
angssr = angss.reshape([nro,na*nxs,nz])

anggne = agc(angssr)

#viewimgframeskey(anggne,transp=True,pclip=0.5)

ntot = anggne.shape[1]
minidx = 10*na; maxidx = int(ntot/2) + 10*na
ixmin = int(minidx/na)*jx; ixmax = int(maxidx/na)*jx

makemovie_mpl(anggne[:,minidx:maxidx,50:400].T,'./fig/angrmig',qc=True,pclip=0.5,ttlstring=r'$\rho$=%.4f',ottl=oro,dttl=dro,
    labelsize=15,hbox=6,wbox=10,ticksize=15,xlabel='X (km)',ylabel='Z (km)',zmin=(100)*dz/1000.0,
    zmax=(300-1)*dz/1000.0,xmin=ixmin*dx/1000.0,xmax=ixmax*dx/1000.0)

