import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt
from scaas.off2ang import off2angkzx, get_angkzx_axis

sep = seppy.sep()

iaxes,img = sep.read_file("spimgbobextw.H")
img = img.reshape(iaxes.n,order='F').T
imgn = img[np.newaxis]
[dz,dx,dy,dhx] = iaxes.d; [oz,ox,oy,ohx] = iaxes.o

ang = off2angkzx(imgn,ohx,dhx,dz,eps=10.0,na=41,transp=True)
na,oa,da = get_angkzx_axis(na=41)

sep.write_file("spimgbobangw.H",ang.T,os=[0,oa,0.0,ox,0.0],ds=[dz,da,1.0,dx,1.0])

