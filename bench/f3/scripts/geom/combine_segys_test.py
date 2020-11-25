import segyio
import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
import matplotlib.pyplot as plt

# Read in the migration cube
sep = seppy.sep()
maxes,mig = sep.read_file("./mig/mig.T")
[nt,nx,ny] = maxes.n; [dt,dx,dy] = maxes.d; [ot,ox,oy] = maxes.o
mig = mig.reshape(maxes.n,order='F')

# Scaling factor
sc = 0.001

# Read in SEGYs
datsgy1 = segyio.open('./segy/SW8103.rode_0001.segy',ignore_geometry=True)
datsgy2 = segyio.open('./segy/SW8113.rode_0001.segy',ignore_geometry=True)

# Read in the data
data1 = datsgy1.trace.raw[:]

# Read in the data
data2 = datsgy2.trace.raw[:]

# Get source coordinates
srcx1 = np.asarray(datsgy1.attributes(segyio.TraceField.SourceX),dtype='int32')
srcy1 = np.asarray(datsgy1.attributes(segyio.TraceField.SourceY),dtype='int32')

srccoords1 = np.zeros([len(srcx1),2],dtype='int')
srccoords1[:,0] = srcy1
srccoords1[:,1] = srcx1

# Get unique coordinates
ucoords1,cts1 = np.unique(srccoords1,axis=0,return_counts=True)

isht = 30
srcx1 = ucoords1[isht,1]*sc; srcy1 = ucoords1[isht,0]*sc

# Get reciver coordinates
recx1 = np.asarray(datsgy1.attributes(segyio.TraceField.GroupX),dtype='int32')
recy1 = np.asarray(datsgy1.attributes(segyio.TraceField.GroupY),dtype='int32')

# Get receivers for that shot
idx1 = srccoords1 == ucoords1[isht]
s = np.sum(idx1,axis=1)
nidx1 = s == 2
recx11 = recx1[nidx1]*sc; recy11 = recy1[nidx1]*sc


srcx2 = np.asarray(datsgy2.attributes(segyio.TraceField.SourceX),dtype='int32')
srcy2 = np.asarray(datsgy2.attributes(segyio.TraceField.SourceY),dtype='int32')

srccoords2 = np.zeros([len(srcx2),2],dtype='int')
srccoords2[:,0] = srcy2
srccoords2[:,1] = srcx2

# Get unique coordinates
ucoords2,cts2 = np.unique(srccoords2,axis=0,return_counts=True)

srcx2 = ucoords2[isht,1]*sc; srcy2 = ucoords2[isht,0]*sc

recx2 = np.asarray(datsgy2.attributes(segyio.TraceField.GroupX),dtype='int32')
recy2 = np.asarray(datsgy2.attributes(segyio.TraceField.GroupY),dtype='int32')

# Get receivers for that shot
idx2 = srccoords2 == ucoords2[isht]
s = np.sum(idx2,axis=1)
nidx2 = s == 2
recx21 = recx2[nidx2]*sc; recy21 = recy2[nidx2]*sc

# Concatenate receivers
recx = np.concatenate([recx11,recx21],axis=0)
recy = np.concatenate([recy11,recy21],axis=0)

# Subtract away origin
ox = 469800.0*sc; oy = 6072300.0*sc
srcx1 -= ox; srcy1 -= oy
recx -= ox; recy -= oy

# Window the data
shts1 = data1[nidx1,:]
shts2 = data2[nidx2,:]
ntr1,nt = shts1.shape
ntr2,nt = shts2.shape
dt = 0.002
datot = np.concatenate([shts1,shts2],axis=0)
ntr,nt = datot.shape

#fig = plt.figure(figsize=(15,15)); ax = fig.gca()
#ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[0,nx*dx,0,ny*dy])
#ax.scatter(srcx1,srcy1,marker='*',color='tab:red')
#ax.scatter(recx,recy,marker='v',color='tab:green')

fig = plt.figure(figsize=(15,15)); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400,700:1000,:100].T)),cmap='gray',extent=[700*dx,1000*dx,0,100*dy])
ax.scatter(srcx1,srcy1,marker='*',color='tab:red')
ax.scatter(recx,recy,marker='v',color='tab:green')
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Y (km)',fontsize=15)
ax.tick_params(labelsize=15)
plt.savefig('./fig/oneshotcoords.png',dpi=150,transparent=True,bbox_inches='tight')

fig = plt.figure(); ax = fig.gca()
pclip = 0.05
dmin = pclip*np.min(datot); dmax = pclip*np.max(datot);
ax.imshow(datot[:,:2000].T,cmap='gray',extent=[0,ntr,1500*dt,0],aspect='auto',vmin=dmin,vmax=dmax)
ax.set_xlabel('Receiver no',fontsize=15)
ax.set_ylabel('Time (s)',fontsize=15)
ax.tick_params(labelsize=15)
plt.savefig('./fig/oneshotdata.png',dpi=150,transparent=True,bbox_inches='tight')

#plt.show()

