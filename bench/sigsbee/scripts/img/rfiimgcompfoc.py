import inpout.seppy as seppy
import numpy as np
from genutils.movie import viewimgframeskey

sep = seppy.sep()

# Defocused stack
saxes,stk = sep.read_file("sigfocrfi2.H")
[nz,nx] = saxes.n; [dz,dx] = saxes.d; [oz,ox] = saxes.o
stk = stk.reshape(saxes.n,order='F')

# Refocused image
raxes,rfi = sep.read_file("stkfocwind2.H")
rfi = rfi.reshape(raxes.n,order='F')

# Correct image
#taxes,tru = sep.read_file("sigsbeeresmsktstkro1.H")
#tru = tru.reshape(taxes.n,order='F')
# Window image
#truw = tru[240:1150,20:480]

comb = np.zeros([2,nz,nx],dtype='float32')
comb[0] = stk; comb[1] = rfi; #comb[2] = truw

viewimgframeskey(comb,interp='bilinear',transp=False,xmin=ox,xmax=ox+nx*dx,
                 zmin=oz,zmax=oz+nz*dz,pclip=0.2)

