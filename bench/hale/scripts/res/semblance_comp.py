import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from scaas.trismooth import smooth
from joblib import Parallel,delayed
from genutils.plot import plot_img2d, plot_anggatrhos
from genutils.movie import viewimgframeskey
import matplotlib.pyplot as plt

# IO
sep = seppy.sep()

# Parameters
rectz,rectro = 15,3
nthreads = 24

# Read in regular gathers
gaxes,gat = sep.read_file("resfaultfocuswindt.H")
dz,da,dx,dro = gaxes.d; oz,oa,ox,oro = gaxes.o
gat = gat.reshape(gaxes.n,order='F').T
gatw = gat[:,:,32:,:]

# Get dimensions
nro,nx,na,nz = gatw.shape
# Compute agc
gangs = np.asarray(Parallel(n_jobs=nthreads)(delayed(agc)(gatw[iro]) for iro in range(nro)))
sep.write_file("gangs.H",gangs.T,ds=[dz,da,dx,dro],os=[oz,0,ox,oro])

# Compute semblance
gstack   = np.sum(gatw,axis=2)
gstackg  = np.sum(gangs,axis=2)
gstacksq = gstackg*gstackg
gnum = smooth(gstacksq.astype('float32'),rect1=rectz,rect3=rectro)

gsqstack = np.sum(gangs*gangs,axis=2)
gden = smooth(gsqstack.astype('float32'),rect1=rectz,rect3=rectro)

gsemb = gnum/gden

gsembt = np.transpose(gsemb,(1,0,2)) # [nro,nx,nz] -> [nx,nro,nz]

gsembt /= np.max(gsembt)

viewimgframeskey(gsembt,cmap='jet',dz=dz,dx=dro,ox=oro,show=False)

# Read in muted gathers
maxes,mut = sep.read_file("resfaultfocuswindtmute.H")
dz,da,dx,dro = maxes.d; oz,oa,ox,oro = maxes.o
mut = mut.reshape(maxes.n,order='F').T
mutw = mut[:,:,32:,:]

# Get dimensions
nro,nx,na,nz = mutw.shape
# Compute agc
mangs = np.asarray(Parallel(n_jobs=nthreads)(delayed(agc)(mutw[iro]) for iro in range(nro)))
sep.write_file("mutw.H",mutw.T,os=[oz,0,ox,oro],ds=[dz,da,dx,dro])
sep.write_file("mangs.H",mangs.T,os=[oz,0,ox,oro],ds=[dz,da,dx,dro])

# Compute semblance
mstack   = np.sum(mutw,axis=2)
mstackg  = np.sum(mangs,axis=2)
mstacksq = mstackg*mstackg
mnum = smooth(mstacksq.astype('float32'),rect1=rectz,rect3=rectro)

msqstack = np.sum(mangs*mangs,axis=2)
mden = smooth(msqstack.astype('float32'),rect1=rectz,rect3=rectro)

msemb = mnum/mden

msembt = np.transpose(msemb,(1,0,2)) # [nro,nx,nz] -> [nx,nro,nz]

msembt /= np.max(msembt)

viewimgframeskey(msembt,cmap='jet',dz=dz,dx=dro,ox=oro,show=True)

# Examine differences
#plot_anggatrhos(gatw,250,dz,dx,oro,dro,imgaspect=4.0,pclip=0.3,show=False)
#plot_anggatrhos(mutw,250,dz,dx,oro,dro,imgaspect=4.0,pclip=0.3,show=True)

#viewimgframeskey(gstack,cmap='gray',dz=dz,dx=dx,ox=ox,show=False)
#viewimgframeskey(gstackg,cmap='gray',dz=dz,dx=dx,ox=ox,show=True)

# Agc'ed stacks
#viewimgframeskey(mstack ,cmap='gray',dz=dz,dx=dx,ox=ox,show=False)
#viewimgframeskey(gstackg-mstackg,cmap='gray',dz=dz,dx=dx,ox=ox,show=False)
#viewimgframeskey(mstackg,cmap='gray',dz=dz,dx=dx,ox=ox,show=True)

# Numerator
#viewimgframeskey(gnum,cmap='jet',dz=dz,dx=dro,ox=oro,show=False)
#viewimgframeskey(mnum,cmap='jet',dz=dz,dx=dro,ox=oro,show=True)


