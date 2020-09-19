import inpout.seppy as seppy
import numpy as np
from scaas.off2ang import off2angkzx, get_angkzx_axis

sep = seppy.sep()

iaxes,img = sep.read_file("sigextimgwrngpos.H")
[nz,nx,ny,nhx] = iaxes.n; [oz,ox,oy,ohx] = iaxes.o; [dz,dx,dy,dhx] = iaxes.d
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')
imgn = img[np.newaxis]

na = 64
ang = off2angkzx(imgn,ohx,dhx,dz,na=na,nthrds=10,transp=True,cverb=True)
na,oa,da = get_angkzx_axis(na,amax=60)

sep.write_file("sigsbee_angoverwpos.H",ang.T,os=[oz,oa,0.0,ox,0.0],ds=[dz,da,1.0,dx,1.0])

