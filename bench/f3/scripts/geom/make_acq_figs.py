import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()
saxes,srcs = sep.read_file("srccoordsall.H")
srcs = srcs.reshape(saxes.n,order='F')
srcy = srcs[:,0]
srcx = srcs[:,1]

usrcs = np.unique(srcs,axis=0)
usrcy = usrcs[:,0]
usrcx = usrcs[:,1]

raxes,recs = sep.read_file("reccoordsall.H")
recs = recs.reshape(raxes.n,order='F')
recy = recs[:,0]
recx = recs[:,1]

urecs = np.unique(recs,axis=0)
urecy = urecs[:,0]
urecx = urecs[:,1]

# Calculate midpoints
mptx = (srcx+recx)/2.0
mpty = (srcy+recy)/2.0

mpts = np.zeros(recs.shape,dtype='float32')
mpts[:,0] = mpty
mpts[:,1] = mptx
umpts = np.unique(mpts,axis=0)
umpty = umpts[:,0]
umptx = umpts[:,1]

# Read in migration cube
maxes,mig = sep.read_file("./mig/mig.T")
nz,nx,ny = maxes.n; oz,ox,oy = maxes.o; dz,dx,dy = maxes.d
dx *= 1000; dy *= 1000;
mig = mig.reshape(maxes.n,order='F')

# Origin
ox = 469800.0; oy = 6072350.0

print("Src extent: %d %d %d %d"%(np.min(srcy),np.min(srcx),np.max(srcy),np.max(srcx)))
print("Rec extent: %d %d %d %d"%(np.min(recy),np.min(recx),np.max(recy),np.max(recx)))
print("Mpt extent: %.2f %.2f %.2f %.2f"%(np.min(mpty),np.min(mptx),np.max(mpty),np.max(mptx)))

fig = plt.figure(figsize=(15,10)); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox,ox+nx*dx,oy,oy+ny*dy])
ax.scatter(usrcx,usrcy,marker='*',color='tab:red')
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.set_title('Source locations: OriginX=469800 OriginY=6072350',fontsize=15)
ax.tick_params(labelsize=15)
plt.savefig('./fig/srclocsall.png',dpi=150,transparent=False,bbox_inches='tight')
plt.close()

# Remove receiver coordinates outside
idx = np.where(urecy < oy)
purecy = np.delete(urecy,idx)
purecx = np.delete(urecx,idx)
fig = plt.figure(figsize=(15,10)); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox,ox+nx*dx,oy,oy+ny*dy])
ax.scatter(purecx,purecy,marker='v',color='tab:green',s=20)
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.tick_params(labelsize=15)
ax.set_title('Receiver locations: OriginX=469800 OriginY=6072350',fontsize=15)
plt.savefig('./fig/reclocsall.png',dpi=150,transparent=False,bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(15,10)); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox,ox+nx*dx,oy,oy+ny*dy])
ax.scatter(umptx,umpty,color='tab:orange',s=20)
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.set_title('Midpoint locations: OriginX=469800 OriginY=6072350',fontsize=15)
ax.tick_params(labelsize=15)
plt.savefig('./fig/mptlocsall.png',dpi=150,transparent=False,bbox_inches='tight')
plt.close()

