import segyio
import glob
import inpout.seppy as seppy
import numpy as np
from genutils.ptyprint import progressbar
import matplotlib.pyplot as plt

# Read in the migration cube
sep = seppy.sep()
maxes,mig = sep.read_file("./mig/mig.H")
[nt,nx,ny] = maxes.n; [dt,dx,dy] = maxes.d; [ot,ox,oy] = maxes.o
mig = mig.reshape(maxes.n,order='F')

# Read in the SEGY file
sgys = glob.glob("./segy/*.segy")

allcoords = []
for isgy in progressbar(range(len(sgys)),"nsgy:"):
  # Read in the SEGY
  datsgy = segyio.open(sgys[isgy],ignore_geometry=True)

  # Get the coordinates
  srcx = np.asarray(datsgy.attributes(segyio.TraceField.SourceX),dtype='int32')
  srcy = np.asarray(datsgy.attributes(segyio.TraceField.SourceY),dtype='int32')
  recx = np.asarray(datsgy.attributes(segyio.TraceField.GroupX),dtype='int32')
  recy = np.asarray(datsgy.attributes(segyio.TraceField.GroupY),dtype='int32')

  srccoords = np.zeros([len(srcx),2],dtype='int')

  srccoords[:,0] = srcy
  srccoords[:,1] = srcx

  srccoords[np.lexsort((srccoords[:,0],srccoords[:,1]))]

  ucoords = np.unique(srccoords,axis=0)

  allcoords.append(ucoords)

allcoords = np.concatenate(allcoords,axis=0).astype('float32')
print(allcoords.shape)

ncrd = allcoords.shape[0]

sep.write_file("srccoords.H",allcoords)

#usrcy = ucoords[:,0]*1E-4
#usrcx = ucoords[:,1]*1E-4

#fig = plt.figure(); ax = fig.gca()
#ax.imshow(np.flipud(mig[150].T),cmap='gray',extent=[ox,ox+nx*dx,oy,oy+ny*dy])
#ax.scatter(usrcx2,usrcy2);
#plt.show()

#TODO: Plot all of the receivers for this one shot

