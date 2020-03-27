import numpy as np
import inpout.seppy as seppy
import h5py

sep = seppy.sep([])

# Get the dimensions
iaxes= sep.read_header(None,ifname='img9896.H')

nfiles = 30
nex = nfiles * iaxes.n[4]
# Output data
nz = iaxes.n[0]; nx = iaxes.n[1]; nro = iaxes.n[3]
trimgs = np.zeros([nz,nx,nro,nex],dtype='float32')

beg = 0; end = 0
for ifile in range(nfiles):
  print(ifile)
  iaxes,img = sep.read_file(None,ifname='img9896.H',safe=False)
  end += iaxes.n[4]
  img = img.reshape(iaxes.n,order='F')
  trimgs[:,:,:,beg:end] = img[:,:,10,:,:]
  beg = end

# Transpose
trimgst = np.transpose(trimgs,(3,0,1,2))
with h5py.File('test.h5','w') as hf:
  hf.create_dataset("data", (nex,nz,nx,nro), data=trimgst, dtype=np.float32)
