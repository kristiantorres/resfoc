import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig, postresmig, get_rho_axis
from scaas.off2ang import off2angkzx, get_angkzx_axis
from genutils.movie import viewimgframeskey
from genutils.plot import plot_img2d

# Read in a single example
sep = seppy.sep()
iaxes,img = sep.read_wind("hale_deftrimgs_off.H",fw=3,nw=1)
dz,dx,dy,dhx,dm = iaxes.d; oz,ox,oy,ohx,om = iaxes.o
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)[:,:,140:652,:450].astype('float32')
imgw = img[:,0,:,:]
zof = img[20,0,:,:]

# Convert to angle
na = 64
ang = off2angkzx(img[np.newaxis],ohx,dhx,dz,na=na,nthrds=20,transp=True)[0,:,0,:,:]
stk = np.sum(ang,axis=1)

# Compare the images
#viewimgframeskey([zof*12,stk],dz=dz,dx=dx)

# Prestack residual migration
inro = 81; idro = 0.001250
pre = preresmig(imgw,[dhx,dx,dz],nps=[65,512,512],nro=inro,dro=idro,nthreads=18,verb=True)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)
prestk = pre[:,20,:,:]

# Poststack residual migration
pst1 = postresmig(zof,[dx,dz],nps=[512,512],nro=inro,dro=idro,nthreads=1,verb=True)
pst2 = postresmig(stk,[dx,dz],nps=[512,512],nro=inro,dro=idro,nthreads=1,verb=True)

viewimgframeskey(pst1,ttlstring=r'Zero-offset $\rho$=%f',dx=dx,dz=dz,ottl=ooro,dttl=odro,show=False)
viewimgframeskey(pst2,ttlstring=r'Angle stack $\rho$=%f',dx=dx,dz=dz,ottl=ooro,dttl=odro,show=False)
viewimgframeskey(prestk,ttlstring=r'$\rho$=%f',dx=dx,dz=dz,ottl=ooro,dttl=odro,show=True)

