import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from deeplearn.utils import normextract, resample, plotseglabel
from deeplearn.focuslabels import corrsim, semblance_power
from deeplearn.dataloader import WriteToH5
import matplotlib.pyplot as plt
from genutils.plot import plot_cubeiso
from genutils.ptyprint import create_inttag, progressbar

# IO
sep = seppy.sep()

taxes = sep.read_header("hale_foctrimgs.H")
[nz,na,naz,nx,nm] = taxes.n
[dz,da,daz,dx,dm] = taxes.d
[oz,oa,oaz,ox,om] = taxes.o

os = [oa,oz,ox]; ds = [da,dz,dx]

# Size of input data for reading
nw = 20; nex = nm//nw
# Size of a single patch
#ptchz = 64; ptchx = 64
ptchz = 128; ptchx = 128

# Define window
bxw = 100;  exw = nx - 100
bzw = 0; ezq = nz

# Open the output HDF5 file
wh5 = WriteToH5("/scr2/joseph29/hale_fltseg128.h5",dsize=1)

k = 0
for iex in progressbar(range(nex),"iex:"):
  # Read in the focused images
  faxes,foc = sep.read_wind("hale_foctrimgs.H",fw=k,nw=nw)
  foc   = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
  foct  = np.ascontiguousarray(np.transpose(foc[:,:,0,:,:],(0,2,1,3)))
  focts = np.sum(foct,axis=1)
  # Read in the labels
  laxes,lbl = sep.read_wind("hale_trlbls.H",fw=k,nw=nw)
  lbl = np.ascontiguousarray(lbl.reshape(laxes.n,order='F').T).astype('float32')
  ldats = []; llbls = []
  for iw in range(nw):
    # Window the image and label and resample
    focw  = focts[iw,bxw:exw,bzw:nz]
    lblw  = lbl  [iw,bxw:exw,bzw:nz]
    focg  = agc(focw)
    # Transpose
    focgt = np.ascontiguousarray(focg.T)
    lblwt = np.ascontiguousarray(lblw.T)
    # Extract patches
    fptch = normextract(focgt,nzp=ptchz,nxp=ptchx,norm=True)
    lptch = normextract(lblwt,nzp=ptchz,nxp=ptchx,norm=False)
    nptch = lptch.shape[0]
    # Write the training data to HDF5 file
    wh5.write_examples(fptch,lptch)
  k += nw

