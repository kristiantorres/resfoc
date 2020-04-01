import inpout.seppy as seppy
import numpy as np
from resfoc import tpow
import matplotlib.pyplot as plt
from utils.movie import viewimgframeskey

sep = seppy.sep()

raxes,rng = sep.read_file("stormangaj3.H")

nz = aaxes.n[0]; na = aaxes.n[1]; nx = aaxes.n[2]; nro = aaxes.n[3]
dz = aaxes.d[0]; da = aaxes.d[1]; dx = aaxes.d[2]; dro = aaxes.d[3]
oro = aaxes.o[3]

# Subsample along spatial axis


#ang = ang.reshape([nz,na*nx,nro],order='F')
#rng = rng.reshape([nz,na*nx,nro],order='F')

#ang = np.transpose(ang,(2,0,1))
#rng = np.transpose(rng,(2,0,1))
#
#viewimgframeskey(ang,transp=False,pclip=0.2,xmax=(nx)*dx/1000.0,zmax=(nz)*dz/1000.0,ttlstring='rho=%.3f',ottl=oro,dttl=dro,show=False)
#viewimgframeskey(rng,transp=False,pclip=0.2,xmax=(nx)*dx/1000.0,zmax=(nz)*dz/1000.0,ttlstring='rho=%.3f',ottl=oro,dttl=dro)



