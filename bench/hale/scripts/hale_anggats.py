import inpout.seppy as seppy
import numpy as np
from genutils.plot import plot_allanggats

sep = seppy.sep()

aaxes,ang = sep.read_file("spimgbobang.H")
[dz,da,daz,dx] = aaxes.d;
ang = ang.reshape(aaxes.n,order='F').T
angw = ang[100:300,0,20:,:600]

plot_allanggats(angw,dz,dx,jx=8,aagc=False,show=True,pclip=0.5,interp='none')



