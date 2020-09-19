import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from deeplearn.utils import normextract
from deeplearn.dataloader import WriteToH5
import matplotlib.pyplot as plt
from genutils.plot import plot_cubeiso
from genutils.ptyprint import progressbar
import time

# IO
sep = seppy.sep()

taxes = sep.read_header("sigsbee_foctrimgs.H")
[nz,na,naz,nx,nm] = taxes.n
[dz,da,daz,dx,dm] = taxes.d
[oz,oa,oaz,ox,om] = taxes.o

os = [oa,oz,ox]; ds = [da,dz,dx]

# Size of input data for reading
nw = 20; nex = nm//nw
# Size of a single patch
ptchz = 64; ptchx = 64

# Define window
bxw = 20;  exw = nx - 20
bzw = 177; ezq = nz

# Open the output HDF5 file
wh5 = WriteToH5("/scr2/joseph29/sigsbee_focdefres.h5",dsize=1)

k = 0
for iex in range(nex):
  print("iex=%d/%d"%(iex,nex))
  beg = time.time()
  # Read in the focused images
  faxes,foc = sep.read_wind("sigsbee_foctrimgs.H",fw=k,nw=nw)
  foc  = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
  foct = np.ascontiguousarray(np.transpose(foc[:,:,0,:,:],(0,2,1,3)))
  # Read in defocused images
  daxes,dfc = sep.read_wind("sigsbee_deftrimgs.H",fw=k,nw=nw)
  dfc  = np.ascontiguousarray(dfc.reshape(daxes.n,order='F').T).astype('float32')
  dfct = np.ascontiguousarray(np.transpose(dfc[:,:,0,:,:],(0,2,1,3)))
  # Read in residually focused images
  raxes,rfc = sep.read_wind("sigsbee_restrimgs.H",fw=k,nw=nw)
  rfc  = np.ascontiguousarray(rfc.reshape(daxes.n,order='F').T).astype('float32')
  rfct = np.ascontiguousarray(np.transpose(rfc[:,:,0,:,:],(0,2,1,3)))
  print("Finished reading: %f"%(time.time() - beg))
  for iw in progressbar(range(nw),"nw:"):
    # Window the image and label and resample
    focw  = foct[iw,:,bxw:exw,bzw:nz]
    dfcw  = dfct[iw,:,bxw:exw,bzw:nz]
    rfcw  = rfct[iw,:,bxw:exw,bzw:nz]
    # Extract patches
    fptch = normextract(focw,nzp=ptchz,nxp=ptchx,norm=True)
    dptch = normextract(dfcw,nzp=ptchz,nxp=ptchx,norm=True)
    rptch = normextract(rfcw,nzp=ptchz,nxp=ptchx,norm=True)
    # Transpose the patches
    fptcht = np.ascontiguousarray(np.transpose(fptch,(0,1,3,2)))
    dptcht = np.ascontiguousarray(np.transpose(dptch,(0,1,3,2)))
    rptcht = np.ascontiguousarray(np.transpose(rptch,(0,1,3,2)))
    # Create image stacks
    fsptch = np.sum(fptcht,axis=1); dsptch = np.sum(dptcht,axis=1); rsptch = np.sum(rptcht,axis=1)
    nptch = fptch.shape[0]
    fdrptch = np.concatenate([fsptch,dsptch,rsptch],axis=0)
    # Write the training data to HDF5 file
    label = np.zeros([fdrptch.shape[0],1],dtype='float32') + -1
    wh5.write_examples(fdrptch,label)
  print("Finished examples %f"%(time.time() - beg))
  k += nw

