import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from genutils.movie import viewimgframeskey

sep = seppy.sep()

# Read in focused images
faxes,foc = sep.read_file("hale_foctrimgs.H")
[nz,na,ny,nx,nmf] = faxes.n; [oz,oa,oy,ox,om] = faxes.o; [dz,da,dy,dx,dm] = faxes.d
foc = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
focstk = np.sum(foc[:,:,0,:,:],axis=2)
gfoc = focstk

# Read in defocused images
daxes,dfc = sep.read_file("hale_deftrimgs.H")
[nz,na,ny,nx,nmd] = daxes.n
dfc = np.ascontiguousarray(dfc.reshape(daxes.n,order='F').T).astype('float32')
dfcstk = np.sum(dfc[:,:,0,:,:],axis=2)
gdfc = dfcstk

# Read in residually defocused images
raxes,res = sep.read_file("hale_restrimgs.H")
res = np.ascontiguousarray(res.reshape(raxes.n,order='F').T).astype('float32')
resstk = np.sum(res[:,:,0,:,:],axis=2)
gres = resstk

tot = np.zeros([2*nmf+nmd,nx,nz],dtype='float32')

tot[0::3,:,:] = gfoc[:]
tot[1::3,:,:] = gdfc[:]
tot[2::3,:,:] = gres[:]

viewimgframeskey(tot,pclip=0.5,interpolation='bilinear',show=True,
                 zmin=0,zmax=nz*dz,xmin=ox,xmax=ox+nx*dx)

