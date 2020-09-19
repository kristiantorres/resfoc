import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from oway.utils import interp_vel
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in velocity
iaxes,img = sep.read_file("sigimgdistr.H")
[nz,ny,nx] = iaxes.n; [dz,dy,dx] = iaxes.d; [oz,oy,ox] = iaxes.o
img = img.reshape(iaxes.n,order='F')

# Read in reflectivity
laxes,lbls = sep.read_file("overwdiff.H")
[nzl,nxl]  = laxes.n; [dzl,dxl] = laxes.d; [ozl,oxl] = laxes.o
lbls = np.ascontiguousarray(lbls.reshape(laxes.n,order='F'))

lblp = np.zeros([nzl,ny,nxl],dtype='float32')
lblp[:,0,:] = lbls

lbli = interp_vel(nz,
                  ny,oy,dy,
                  nx,ox,dx,
                  lblp,dxl,dy,oxl,oy)

# Write out interpolated reflectivity
lbliw  = lbli[:,0,:]
sep.write_file("overwinterp.H",lbliw,os=[oz,ox],ds=[dz,dx])

