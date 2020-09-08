import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from genutils.movie import viewimgframeskey

sep = seppy.sep()

# Read in focused images
faxes,foc = sep.read_file("sigsbee_foctrimgs.H")
[nz,na,ny,nx,nmf] = faxes.n; [oz,oa,oy,ox,om] = faxes.o; [dz,da,dy,dx,dm] = faxes.d
foc = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
focstk = np.sum(foc[:,:,0,:,:],axis=2)
gfoc = agc(focstk)

# Read in defocused images
daxes,dfc = sep.read_file("sigsbee_deftrimgs.H")
[nz,na,ny,nx,nmd] = daxes.n
dfc = np.ascontiguousarray(dfc.reshape(daxes.n,order='F').T).astype('float32')
dfcstk = np.sum(dfc[:,:,0,:,:],axis=2)
gdfc = agc(dfcstk)

tot = np.zeros([nmf+nmd,nx,nz],dtype='float32')

tot[0::2,:,:] = gfoc[:]
tot[1::2,:,:] = gdfc[:]

viewimgframeskey(tot,pclip=0.5,interpolation='bilinear',show=True,
                 zmin=0,zmax=nz*dz,xmin=ox,xmax=ox+nx*dx)

