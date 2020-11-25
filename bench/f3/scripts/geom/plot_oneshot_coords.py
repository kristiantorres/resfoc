import segyio
import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
import matplotlib.pyplot as plt
from genutils.ptyprint import progressbar

# Read in the migration cube
sep = seppy.sep()
maxes,mig = sep.read_file("./mig/mig.T")
[nt,nx,ny] = maxes.n; [dt,dx,dy] = maxes.d; [ot,ox,oy] = maxes.o
mig = mig.reshape(maxes.n,order='F')

# Scaling factor
sc = 0.001

# Read in SEGY
datsgy = segyio.open('./segy/SW8101.rode_0001.segy',ignore_geometry=True)

# Get source coordinates
srcx = np.asarray(datsgy.attributes(segyio.TraceField.SourceX),dtype='int32')
srcy = np.asarray(datsgy.attributes(segyio.TraceField.SourceY),dtype='int32')

srccoords = np.zeros([len(srcx),2],dtype='int')
srccoords[:,0] = srcy
srccoords[:,1] = srcx

# Get unique coordinates
ucoords,cts = np.unique(srccoords,axis=0,return_counts=True)
nsht = ucoords.shape[0]

# Get reciver coordinates
recx = np.asarray(datsgy.attributes(segyio.TraceField.GroupX),dtype='int32')
recy = np.asarray(datsgy.attributes(segyio.TraceField.GroupY),dtype='int32')

# Loop over each unique source
jsht = 1
for isht in progressbar(range(nsht),"nsht:",verb=True):
  if(isht%jsht == 0):
    # Get receivers for that shot
    idx = srccoords == ucoords[isht]
    s = np.sum(idx,axis=1)
    nidx = s == 2
    recx1 = recx[nidx]*sc; recy1 = recy[nidx]*sc

    srcx1 = ucoords[isht,1]*sc; srcy1 = ucoords[isht,0]*sc

    # Subtract away origin
    ox = 469800.0*sc; oy = 6072299.0*sc
    srcx1 -= ox; srcy1 -= oy
    recx1 -= ox; recy1 -= oy

    fig = plt.figure(figsize=(15,15)); ax = fig.gca()
    ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[0,nx*dx,0,ny*dy])
    ax.scatter(srcx1,srcy1,marker='*',color='tab:red')
    ax.scatter(recx1,recy1,marker='v',color='tab:green')
    plt.show()

