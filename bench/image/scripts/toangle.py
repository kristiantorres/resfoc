import numpy as np
import inpout.seppy as seppy
from scaas.off2ang import off2ang, get_ang_axis
from resfoc.resmig import preresmig,get_rho_axis
from resfoc import tpow
from utils.movie import viewimgframeskey

sep = seppy.sep([])

iaxes,img = sep.read_file(None,"bigresmig.H")
img = img.reshape(iaxes.n,order='F')

nz = iaxes.n[0]; nx = iaxes.n[1]; nh = iaxes.n[2]; nro = iaxes.n[3]
dz = iaxes.d[0]; dx = iaxes.d[1]; dh = iaxes.d[2]; dro = iaxes.d[3]
oz = iaxes.d[0]; ox = iaxes.d[1]; oh = iaxes.o[2]; oro = iaxes.o[3]

imgt = np.ascontiguousarray(np.transpose(img,(3,2,0,1))).astype('float32')

storma = off2ang(imgt,oh,dh,dz,nthrds=24,oro=oro,dro=dro,verb=True)
na,oa,da = get_ang_axis()

print(storma.shape)

viewimgframeskey(storma[:,512,:,:],pclip=0.2,interp='sinc')

stormat = np.transpose(storma,(3,2,1,0))
aaxes = seppy.axes([nz,na,nx,nro],[oz,oa,ox,oro],[dz,da,dx,dro])
sep.write_file(None,aaxes,stormat,ofname='stormang.H')

