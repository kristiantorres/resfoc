import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()
saxes,srcs = sep.read_file("srccoordsall.H")
srcs = srcs.reshape(saxes.n,order='F')
srcy = srcs[:,0]
srcx = srcs[:,1]

raxes,recs = sep.read_file("reccoordsall.H")
recs = recs.reshape(raxes.n,order='F')
recy = recs[:,0]
recx = recs[:,1]

mptx = (srcx+recx)/2.0
mpty = (srcy+recy)/2.0

# Read in migration cube
maxes,mig = sep.read_file("./mig/mig.T")
nz,nx,ny = maxes.n; oz,ox,oy = maxes.o; dz,dx,dy = maxes.d
dx *= 1000; dy *= 1000;
mig = mig.reshape(maxes.n,order='F')

# Migration cube origin
ox1 = 469800.0; oy1 = 6072300.0
ox1c = np.min(mptx); oy1c = np.min(mpty)
ox2 = 469800.0; oy2 = 6072350.0

print("Src extent: %d %d %d %d"%(np.min(srcy),np.min(srcx),np.max(srcy),np.max(srcx)))
print("Rec extent: %d %d %d %d"%(np.min(recy),np.min(recx),np.max(recy),np.max(recx)))
print("Mpt extent: %.2f %.2f %.2f %.2f"%(np.min(mpty),np.min(mptx),np.max(mpty),np.max(mptx)))

fig = plt.figure(figsize=(15,10)); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox1,ox1+nx*dx,oy1c,oy1c+ny*dy])
ax.scatter(mptx[::1000],mpty[::1000],color='tab:blue')
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.set_title('My shifted origin: X=469800 Y=6072299',fontsize=15)
ax.tick_params(labelsize=15)

fig = plt.figure(figsize=(15,10)); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox2,ox2+nx*dx,oy2,oy2+ny*dy])
ax.scatter(mptx[::2000],mpty[::2000],color='tab:blue')
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.set_title('Origin from EBCDIC: X=469800 Y=6071325',fontsize=15)
ax.tick_params(labelsize=15)

plt.show()

