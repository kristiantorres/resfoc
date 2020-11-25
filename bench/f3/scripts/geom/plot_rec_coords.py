import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()
# Read in the source coordinates
raxes,coords = sep.read_file('reccoordsall.H')
coords = coords.reshape(raxes.n,order='F')

ucoords = np.unique(coords,axis=0)

sc = 0.001

recy = coords[:,0]*sc
recx = coords[:,1]*sc

# Read in migration cube
maxes,mig = sep.read_file("./mig/mig.T")
nz,nx,ny = maxes.n; oz,ox,oy = maxes.o; dz,dx,dy = maxes.d
mig = mig.reshape(maxes.n,order='F')

# Migration cube origin
ox = 469800.0*sc; oy = 6072325.0*sc

recx -= ox; recy -= oy

print(np.min(recy),np.min(recx),np.max(recy),np.max(recx))

fig = plt.figure(); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[0,nx*dx,0,ny*dy])
ax.scatter(recx[::20],recy[::20])
plt.show()

