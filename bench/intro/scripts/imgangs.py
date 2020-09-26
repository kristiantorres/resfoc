import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt
from genutils.plot import plot_imgpang

sep = seppy.sep()

iaxes,img = sep.read_file("intro_resang.H")
img = img.reshape(iaxes.n,order='F').T
[nz,na,nx,nro] = iaxes.n; [dz,da,dx,dro] = iaxes.d; [oz,oa,ox,oro] = iaxes.o

amin = np.min(img); amax = np.max(img)
stk = np.sum(img,axis=2)
imin = np.min(stk); imax = np.max(stk)

ros = np.linspace(oro,oro+(nro-1)*dro,nro)

print(ros[20],ros[12],ros[28])

ro1 = img[20]
rot1 = np.transpose(ro1,(1,2,0))
plot_imgpang(rot1,dx,dz,512,oa,da,wspace=-0.3,aaspect=75,hbox=5,
             imin=imin,imax=imax,amin=amin,amax=amax,plotline=False,
             figname='./fig/focsangnoline.png')

roo = img[12]
root = np.transpose(roo,(1,2,0))
plot_imgpang(root,dx,dz,512,oa,da,wspace=-0.3,aaspect=75,hbox=5,
             imin=imin,imax=imax,amin=amin,amax=amax,plotline=False,
             figname='./fig/overangnoline.png')

rol = img[28]
rolt = np.transpose(rol,(1,2,0))
plot_imgpang(rolt,dx,dz,512,oa,da,wspace=-0.3,aaspect=75,hbox=5,
             imin=imin,imax=imax,amin=amin,amax=amax,plotline=False,
             figname='./fig/underangnoline.png')
