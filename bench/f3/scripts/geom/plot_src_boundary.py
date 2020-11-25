import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()
# Read in the source coordinates
saxes,coords = sep.read_file('srccoordsunq.H')
coords = coords.reshape(saxes.n,order='F')

ucoords = np.unique(coords,axis=0)

srcy = ucoords[:,0]
srcx = ucoords[:,1]

# Take the sources only at the boundaries
minsx = np.min(srcx); maxsx = np.max(srcx)
minsy = np.min(srcy); maxsy = np.max(srcy)

minsxg = minsx + 2000
maxsxl = maxsx - 2000
maxsyl = maxsy - 2000
minsyg = minsy + 2000
srcxbnd = []; srcybnd = []
npts = ucoords.shape[0]
for icrd in range(npts):
  if(srcx[icrd] > maxsxl or srcx[icrd] < minsxg):
    srcxbnd.append(srcx[icrd]); srcybnd.append(srcy[icrd])
  if(srcy[icrd] > maxsyl or srcy[icrd] < minsyg):
    srcxbnd.append(srcx[icrd]); srcybnd.append(srcy[icrd])

# Read in migration cube
maxes,mig = sep.read_file("./mig/mig.T")
nz,nx,ny = maxes.n; oz,ox,oy = maxes.o; dz,dx,dy = maxes.d
dx *= 1000; dy *= 1000;
mig = mig.reshape(maxes.n,order='F')

# Migration cube origin
ox = 469800.0; oy = 6072300.0
#ox = 469800.0; oy = 6071325.0

print("Src extent: %d %d %d %d"%(minsy,minsx,maxsy,maxsx))

fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox,ox+nx*dx,oy,oy+ny*dy])
#ax.scatter(srcx,srcy,marker='*',color='tab:red')
ax.scatter(srcxbnd[::10],srcybnd[::10],marker='*',color='tab:red')
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.tick_params(labelsize=15)

plt.show()

