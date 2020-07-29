import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt
from scaas.off2ang import off2ang, get_ang_axis

sep = seppy.sep()

iaxes,img = sep.read_file("spimgbobfullext.H")
img = img.reshape(iaxes.n,order='F')[:,:,0,:]
imgt = np.ascontiguousarray(np.transpose(img,(2,0,1))).astype('float32') # [nz,nx,nhx] -> [nhx,nz,nx]
[dz,dx,dy,dhx] = iaxes.d; [oz,ox,oy,ohx] = iaxes.o

ang = off2ang(imgt,ohx,dhx,dz,transp=False,verb=True,nthrds=24,ota=-0.02,na=140,dta=0.01,nta=601,amax=70)
na,oa,da = get_ang_axis(amax=60)

sep.write_file("spimgbobfullang.H",ang.T,os=[0,oa,ox],ds=[dz,da,dx])
