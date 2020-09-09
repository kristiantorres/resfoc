import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from oway.utils import interp_vel
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in velocity
iaxes,img = sep.read_file("sigimg.H")
[nz,ny,nx] = iaxes.n; [dz,dy,dx] = iaxes.d; [oz,oy,ox] = iaxes.o
img = img.reshape(iaxes.n,order='F')

# Read in reflectivity
raxes,refs = sep.read_file("sigsbee_trrefs.H")
[nzr,nxr,nm]  = raxes.n; [dzr,dxr,dm] = raxes.d; [ozr,oxr,om] = raxes.o
refs = np.ascontiguousarray(np.transpose(refs.reshape(raxes.n,order='F').T,(0,2,1)))

refp = np.zeros([nm,nzr,ny,nxr],dtype='float32')
refp[:,:,0,:] = refs[:,:,:]

refi = np.zeros([nm,nz,ny,nx],dtype='float32')

for im in range(nm):
  refi[im] = interp_vel(nz,
                        ny,oy,dy,
                        nx,ox,dx,
                        refp[im],dxr,dy,oxr,oy)

# Write out interpolated reflectivity
refiw  = refi[:,:,0,:]
refiwt = np.transpose(refiw,(0,2,1))
sep.write_file("sigsbee_trrefsint.H",refiwt.T,os=[oz,ox,om],ds=[dz,dx,dm])

