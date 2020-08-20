import inpout.seppy as seppy
import numpy as np
from genutils.movie import resangframes

sep = seppy.sep()

aaxes,ang = sep.read_file("halerest.H")
[dz,da,dx,dro] = aaxes.d; [oz,oa,ox,oro] = aaxes.o
ang = ang.reshape(aaxes.n,order='F').T
angw = ang[:,:,20:,:600]

resangframes(angw,dz,dx,dro,oro,jx=10,pclip=0.5,interp='none')

