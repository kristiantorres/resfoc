"""
Creates the regular migrated cube from the
./mig/Z3NAM1989E_Migration.sgy file

@author: Joseph Jennings
@version: 2020.09.27
"""
import segyio
import inpout.seppy as seppy
import numpy as np

migsgy = segyio.open("./mig/Z3NAM1989E_Migration.sgy",ignore_geometry=False)

il3d = np.asarray(migsgy.attributes(segyio.TraceField.INLINE_3D),dtype='int32')
xl3d = np.asarray(migsgy.attributes(segyio.TraceField.CROSSLINE_3D),dtype='int32')

ui,ci = np.unique(il3d,return_counts=True)
ux,cx = np.unique(xl3d,return_counts=True)

ny = len(ui)
nx = len(ux)

# Read in input traces
mig = migsgy.trace.raw[:]
[ntr,nt] = mig.shape

# Output migration cube
migout = np.zeros([ny,nx,nt],dtype='float32')

beg = 0; end = nx
for iy in range(ny):
  migout[iy,:,:] = mig[beg:end,:]
  beg = end; end += nx

# Output axes
ds = [0.004,0.025,0.025] # dt,dx,dy
os = [0.0,46.98,607.325] # ot,ox,oy

sep = seppy.sep()
sep.write_file("./mig/mig.H",migout.T,ds=ds,os=os,dpath="/net/brick5/data3/northsea_dutch_f3/mig/")

