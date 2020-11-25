import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()
# Read in the source coordinates
saxes,scoords = sep.read_file('srccoordsunq.H')
scoords = scoords.reshape(saxes.n,order='F')
# Read in the receiver coordinates
raxes,rcoords = sep.read_file('reccoordsall.H')
rcoords = rcoords.reshape(raxes.n,order='F')

uscoords = np.unique(scoords,axis=0)
srcy = uscoords[:,0]
srcx = uscoords[:,1]

urcoords = np.unique(rcoords,axis=0)
recy = urcoords[:,0]
recx = urcoords[:,1]

# Read in migration cube
maxes,mig = sep.read_file("./mig/mig.T")
nz,nx,ny = maxes.n; oz,ox,oy = maxes.o; dz,dx,dy = maxes.d
dx *= 1000; dy *= 1000;
mig = mig.reshape(maxes.n,order='F')

# Migration cube origin
ox1 = 469800.0; oy1 = 6072299.0
#ox2 = 469800.0; oy2 = 6071325.0
#ox2 = 469800.0; oy2 = 6072351.0
ox2 = 469775; oy2 =  6069875
#ox2 = 469800.0; oy2 = 6072350.0

print("Src extent: %d %d %d %d"%(np.min(srcy),np.min(srcx),np.max(srcy),np.max(srcx)))
print("Rec extent: %d %d %d %d"%(np.min(recy),np.min(recx),np.max(recy),np.max(recx)))

fig = plt.figure(figsize=(15,10)); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox1,ox1+nx*dx,oy1,oy1+ny*dy])
ax.scatter(srcx[::10],srcy[::10],marker='*',color='tab:red')
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.set_title('My shifted origin: X=469800 Y=6072299',fontsize=15)
ax.tick_params(labelsize=15)
#plt.savefig('./fig/grid1_src.png',dpi=150,transparent=False,bbox_inches='tight')
#plt.close()

fig = plt.figure(figsize=(15,10)); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox1,ox1+nx*dx,oy1,oy1+ny*dy])
ax.scatter(recx[::2000],recy[::2000],marker='v',color='tab:green',s=20)
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.tick_params(labelsize=15)
ax.set_title('My shifted origin: X=469800 Y=6072299',fontsize=15)
#plt.savefig('./fig/grid1_rec.png',dpi=150,transparent=False,bbox_inches='tight')
#plt.close()

fig = plt.figure(figsize=(15,10)); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox2,ox2+nx*dx,oy2,oy2+ny*dy])
ax.scatter(srcx[::10],srcy[::10],marker='*',color='tab:red')
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.tick_params(labelsize=15)
ax.set_title('Origin from EBCDIC: X=469800 Y=6071325',fontsize=15)
#plt.savefig('./fig/grid2_src.png',dpi=150,transparent=False,bbox_inches='tight')
#plt.close()

fig = plt.figure(figsize=(15,10)); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox2,ox2+nx*dx,oy2,oy2+ny*dy])
ax.scatter(recx[::2000],recy[::2000],marker='v',color='tab:green',s=20)
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.tick_params(labelsize=15)
ax.set_title('Origin from EBCDIC: X=469800 Y=6071325',fontsize=15)
#plt.savefig('./fig/grid2_rec.png',dpi=150,transparent=False,bbox_inches='tight')
#plt.close()

plt.show()
