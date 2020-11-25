import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()
# Read in the source coordinates
#saxes,coords = sep.read_file('srccoords.H')
#saxes,coords = sep.read_file('srccoordsall.H')
saxes,coords = sep.read_file('srccoordsunq.H')
coords = coords.reshape(saxes.n,order='F')

ucoords = np.unique(coords,axis=0)

srcy = ucoords[:,0]
srcx = ucoords[:,1]

# Read in migration cube
maxes,mig = sep.read_file("./mig/mig.T")
nz,nx,ny = maxes.n; oz,ox,oy = maxes.o; dz,dx,dy = maxes.d
dx *= 1000; dy *= 1000;
mig = mig.reshape(maxes.n,order='F')

# Migration cube origin
ox = 469800.0; oy = 6072299.0
#ox = 469800.0; oy = 6071325.0

print("Src extent: %d %d %d %d"%(np.min(srcy),np.min(srcx),np.max(srcy),np.max(srcx)))

fig = plt.figure(); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox,ox+nx*dx,oy,oy+ny*dy])
ax.scatter(srcx[::10],srcy[::10],marker='*',color='tab:red')
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.tick_params(labelsize=15)

bg1 = np.min(srcx); eg1 = np.max(srcx); dg1 = 0.025
xticks = np.arange(bg1,eg1,dg1)
print(len(xticks),bg1,dg1)
bg2 = np.min(srcy); eg2 = np.max(srcy); dg2 = 0.05
yticks = np.arange(bg2,eg2,dg2)
print(len(yticks),bg2,dg2)
#ax.set_xticks(xticks)
#ax.set_yticks(yticks)
#plt.rc('grid',linestyle="-",color='black')
#plt.grid()
plt.show()

