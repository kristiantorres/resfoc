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
sc = 0.1
imin = sc*np.min(img); imax = sc*np.max(img)

# Read in reflectivity
raxes,ref = sep.read_file("./dat/ref.H",form='native')
[nzr,nxr] = raxes.n; [dzr,dxr] = raxes.d; [ozr,oxr] = raxes.o
ref = ref.reshape(raxes.n,order='F')
nxi = 500; nzi = 1201
refre = resample(ref,[nzi,nxi],kind='linear')
sc = 0.1
rmin = sc*np.min(refre); rmax = sc*np.max(refre)

refp = np.zeros([nzr,ny,nxr],dtype='float32')
refp[:,0,:] = ref


refi = interp_vel(nz,
                  ny,oy,dy,
                  nx,ox,dx,
                  refp,dxr,dy,oxr,oy)

plt.figure()
plt.imshow(refi[:,0,:],cmap='gray',interpolation='bilinear',aspect='auto',vmin=rmin,vmax=rmax)
plt.figure()
plt.imshow(ref,cmap='gray',interpolation='none',aspect='auto',vmin=rmin,vmax=rmax)
plt.figure()
plt.imshow(img[:,0,:],cmap='gray',interpolation='bilinear',aspect='auto',vmin=imin,vmax=imax)
plt.show()

# Write out interpolated reflectivity
sep.write_file("sigsbee_ref.H",refi[:,0,:],os=[oz,ox],ds=[dz,dx])

