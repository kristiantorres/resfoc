import inpout.seppy as seppy
import numpy as np
from resfoc.estro import anglemask
from genutils.movie import viewcube3d
from genutils.plot import plot_img2d

sep = seppy.sep()

saxes,storm = sep.read_file("resmigtrnt.H")
[nz,na,nx,nro] = saxes.n; [oz,oa,ox,oro] = saxes.o; [dz,da,dx,dro] = saxes.d
storm = storm.reshape(saxes.n,order='F').T

# Create the mask
mask = anglemask(nz,na,zpos=0.0,apos=0.4,mode='vert',rand=False,rectz=10,recta=10,verb=True).T

plot_img2d(mask.T)

# Replicate along the spatial dimension
maskrep = np.repeat(mask[np.newaxis,:,:],nx,axis=0)

# Apply the mask to each rho
masked = np.asarray([maskrep*storm[iro] for iro in range(nro)])

# Add small random noise
masked += (np.random.rand(*masked.shape)*2-1)/10000.0

# Write the masked data to file
sep.write_file('resmigtrntmute.H',masked.T,ds=saxes.d,os=saxes.o)

