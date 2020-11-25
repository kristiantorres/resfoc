import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()
# Read in the source coordinates
xaxes,sxcrd = sep.read_file("sepsxwind.H")
yaxes,sycrd = sep.read_file("sepsywind.H")
xaxes,rxcrd = sep.read_file("seprxwind.H")
yaxes,rycrd = sep.read_file("seprywind.H")
npt = len(sxcrd)
scoords = np.zeros([npt,2])
scoords[:,0] = sycrd
scoords[:,1] = sxcrd

rcoords = np.zeros([npt,2])
rcoords[:,0] = rycrd
rcoords[:,1] = rxcrd

# 6072299, new y origin (gives positive for receivers and sources)
uscoords = np.unique(scoords,axis=0)
urcoords = np.unique(rcoords,axis=0)

srcy = uscoords[:,0]
srcx = uscoords[:,1]

recy = urcoords[:,0]
recx = urcoords[:,1]

# Read in migration cube
maxes,mig = sep.read_file("./mig/mig.T")
nz,nx,ny = maxes.n; oz,ox,oy = maxes.o; dz,dx,dy = maxes.d
mig = mig.reshape(maxes.n,order='F')

fig = plt.figure(); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[0,nx*dx,0,ny*dy])
ax.scatter(srcx,srcy)
ax.scatter(recx,recy)

plt.show()
