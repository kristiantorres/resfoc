import inpout.seppy as seppy
import numpy as np

sep = seppy.sep()

# Read in velocity models
vaxes,vels = sep.read_file("trvelswin.H")
vels = np.ascontiguousarray(vels.reshape(vaxes.n,order='F').T)
[oz,ox,om] = vaxes.o; [dz,dx,dm] = vaxes.d

# Split the velocity models
vels1 = vels[0]
vels2 = vels[1:]

## Write the first model
#sep.write_file("velsappend.H",vels1.T,os=[oz,ox],ds=[dz,dx])
#
## Append the remaining
#sep.append_file("velsappend.H",vels2.T)

for im in range(vels.shape[0]):
  if(im == 0):
    sep.write_file("velsappend2.H",vels[0].T,os=[oz,ox],ds=[dz,dx])
  elif(im == 1):
    sep.append_file("velsappend2.H",vels[im].T,newaxis=True)
  else:
    sep.append_file("velsappend2.H",vels[im].T)

