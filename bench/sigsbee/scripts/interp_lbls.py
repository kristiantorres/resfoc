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
laxes,lbls = sep.read_file("sigsbee_trlbls.H")
[nzl,nxl,nm]  = laxes.n; [dzl,dxl,dm] = laxes.d; [ozl,oxl,om] = laxes.o
lbls = np.ascontiguousarray(np.transpose(lbls.reshape(laxes.n,order='F').T,(0,2,1)))

lblp = np.zeros([nm,nzl,ny,nxl],dtype='float32')
lblp[:,:,0,:] = lbls[:,:,:]

lbli = np.zeros([nm,nz,ny,nx],dtype='float32')

for im in range(nm):
  lbli[im] = interp_vel(nz,
                        ny,oy,dy,
                        nx,ox,dx,
                        lblp[im],dxl,dy,oxl,oy)

# Write out interpolated reflectivity
lbliw  = lbli[:,:,0,:]
lbliwt = np.transpose(lbliw,(0,2,1))
sep.write_file("sigsbee_trlblsint.H",lbliwt.T,os=[oz,ox,om],ds=[dz,dx,dm])

