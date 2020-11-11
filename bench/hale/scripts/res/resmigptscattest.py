import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig, postresmig, get_rho_axis
from genutils.movie import viewimgframeskey

sep = seppy.sep()
iaxes,img= sep.read_file("ptscat.H")
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)[:,0,:,:].astype('float32')
dz,dx,dy,dhx = iaxes.d; oz,ox,oy,ohx = iaxes.o

viewimgframeskey(img)

# Prestack Residual migration
inro = 21; idro = 0.01
pre = preresmig(img,[dhx,dx,dz],nps=[65,1024,512],nro=inro,dro=idro,nthreads=18,verb=True)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

zo = img[20,:,:]
pst = postresmig(zo,[dx,dz],nps=[1024,512],nro=inro,dro=idro,nthreads=1,verb=True)

viewimgframeskey(pre[:,20,:,:],show=False,ttlstring=r'Prestack $\rho$=%f',dttl=odro,ottl=ooro)
viewimgframeskey(pst,ttlstring=r'Poststack $\rho$=%f',dttl=odro,ottl=ooro)


