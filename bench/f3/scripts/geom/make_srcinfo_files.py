import os,glob
import segyio
import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
import matplotlib.pyplot as plt
from genutils.ptyprint import progressbar

# Get SEGYs
sgys = sorted(glob.glob('./segy/*.segy'))
nsgy = len(sgys)

# Directory for writing info files
infodir = './segy/info/'

# Migration slice for plotting source coordinates
sep = seppy.sep()
maxes,mig = sep.read_file("./mig/mig.T")
[nt,nx,ny] = maxes.n; [dt,dx,dy] = maxes.d; [ot,ox,oy] = maxes.o
dx *= 1000; dy *= 1000
mig = mig.reshape(maxes.n,order='F')

# Scale factor
sc = 0.001

# Imaging origin
ox = 469800.0; oy = 6072350.0

for isgy in progressbar(range(nsgy),"nsgy:"):
  # Read in the SEGY
  datsgy = segyio.open(sgys[isgy],ignore_geometry=True)

  # Get the coordinates
  srcx = np.asarray(datsgy.attributes(segyio.TraceField.SourceX),dtype='int32')
  srcy = np.asarray(datsgy.attributes(segyio.TraceField.SourceY),dtype='int32')

  srccoords = np.zeros([len(srcx),2],dtype='int')

  srccoords[:,0] = srcy
  srccoords[:,1] = srcx

  ucoords,cts = np.unique(srccoords,axis=0,return_counts=True)
  #ucoords = ucoords.astype('float32')
  #ucoords[:] *= sc
  #ucoords[:,0] -= oy; ucoords[:,1] -= ox
  nsht = ucoords.shape[0]

  # Plot the source coordinates
  fig = plt.figure(figsize=(10,10)); ax = fig.gca()
  ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[ox,ox+nx*dx,oy,oy+ny*dy])
  ax.scatter(ucoords[:,1],ucoords[:,0],marker='*',color='tab:red')
  ax.set_xlabel('X (km)',fontsize=15)
  ax.set_ylabel('Y (km)',fontsize=15)
  ax.tick_params(labelsize=15)
  plt.show()

  ## Loop over unique coordinates and write to file
  #bname = os.path.basename(sgys[isgy])
  #fname = os.path.splitext(bname)[0]
  #ofile = infodir + fname + '.txt'
  #with open(ofile,'w') as f:
  #  for icrd in range(nsht):
  #    f.write('%d %d %d\n'%(ucoords[icrd,0],ucoords[icrd,1],cts[icrd]))

