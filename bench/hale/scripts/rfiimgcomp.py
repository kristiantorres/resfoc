import inpout.seppy as seppy
import numpy as np
from genutils.movie import viewimgframeskey

sep = seppy.sep()

# Defocused stack
saxes,stk = sep.read_file("halestk.H")
[nz,nx] = saxes.n; [dz,dx] = saxes.d; [oz,ox] = saxes.o
stk = stk.reshape(saxes.n,order='F')

# Refocused image
raxes,rfi = sep.read_file("halerfi.H")
rfi = rfi.reshape(raxes.n,order='F')

comb = np.zeros([2,nz,nx],dtype='float32')
comb[0] = stk; comb[1] = rfi

viewimgframeskey(comb[:,:850,:],interp='bilinear',transp=False,xmin=ox,xmax=ox+nx*dx,
                 zmin=oz,zmax=oz+nz*dz,pclip=0.5)

