import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from deeplearn.utils import normextract, resample, plotseglabel
from deeplearn.fltsegpatchchunkr import fltsegpatchchunkr
from deeplearn.dataloader import WriteToH5
from client.sshworkers import create_host_list, launch_sshworkers, kill_sshworkers
from server.utils import startserver
from server.distribute import dstr_collect
from genutils.ptyprint import progressbar, create_inttag
from deeplearn.utils import plotseglabel
import time
import matplotlib.pyplot as plt

# IO
sep = seppy.sep()

taxes = sep.read_header("sigsbee_foctrimgs.H")
[nz,na,naz,nx,nm] = taxes.n
[dz,da,daz,dx,dm] = taxes.d
[oz,oa,oaz,ox,om] = taxes.o

# Size of input data for reading
#nw = 10; nex = nm//nw
nw = 10; nex = 3
# Size of a single patch
nzp = 64; nxp = 64
strdz = int(nzp/2 + 0.5)
strdx = int(nxp/2 + 0.5)

# Define window
bxw = 50;  exw = nx - 50
bzw = 177; ezq = nz

# Create the data writing objects
wh5fseg = WriteToH5('/scr2/joseph29/sigsbee_focseg.h5',dsize=1)
wh5dseg = WriteToH5('/scr2/joseph29/sigsbee_defseg.h5',dsize=1)
wh5rseg = WriteToH5('/scr2/joseph29/sigsbee_resseg.h5',dsize=1)
wh5seg = [wh5fseg,wh5dseg,wh5rseg]

wh5foc = WriteToH5('/scr2/joseph29/sigsbee_fltfoc.h5',dsize=1)
wh5def = WriteToH5('/scr2/joseph29/sigsbee_fltdef.h5',dsize=1)
wh5res = WriteToH5('/scr2/joseph29/sigsbee_fltres.h5',dsize=1)
wh5foc = [wh5foc,wh5def,wh5res]

# Focus labels
focval  = [1,-1,0]

k = 0;
for iex in progressbar(range(nex),"iex:"):
  #TODO: apply agc to each image
  # Read in the images
  beg = time.time()
  faxes,foc = sep.read_wind("sigsbee_foctrimgs.H",fw=k,nw=nw)
  foc   = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
  foct  = np.ascontiguousarray(np.transpose(foc[:,:,0,:,:],(0,2,1,3))) # [nw,nx,na,nz] -> [nw,na,nx,nz]
  # Read in defocused images
  daxes,dfc = sep.read_wind("sigsbee_deftrimgs.H",fw=k,nw=nw)
  dfc   = np.ascontiguousarray(dfc.reshape(daxes.n,order='F').T).astype('float32')
  dfct  = np.ascontiguousarray(np.transpose(dfc[:,:,0,:,:],(0,2,1,3)))
  # Read in residually focused images
  raxes,rfc = sep.read_wind("sigsbee_restrimgs.H",fw=k,nw=nw)
  rfc   = np.ascontiguousarray(rfc.reshape(daxes.n,order='F').T).astype('float32')
  rfct  = np.ascontiguousarray(np.transpose(rfc[:,:,0,:,:],(0,2,1,3)))
  # Read in the labels
  laxes,lbl = sep.read_wind("sigsbee_trlblsint.H",fw=k,nw=nw)
  lbl = np.ascontiguousarray(lbl.reshape(laxes.n,order='F').T).astype('float32')
  #print("Finished reading %f"%(time.time() - beg))
  beg = time.time()
  # Window the data
  focw  = foct[:,:,bxw:exw,bzw:nz]
  dfcw  = dfct[:,:,bxw:exw,bzw:nz]
  rfcw  = rfct[:,:,bxw:exw,bzw:nz]
  lblw  = lbl [:,  bxw:exw,bzw:nz]
  # Concatenate the images
  fdrptch = [focw,dfcw,rfcw]
  lblptch = [lblw,lblw,np.zeros(lblw.shape,dtype='float32')]
  for ity in range(len(fdrptch)):
    ofoc = []; olbl = []; oseg = []
    for j in range(focw.shape[0]):
      datt = np.ascontiguousarray(np.transpose(fdrptch[ity][j],(0,2,1)))
      lblt = np.ascontiguousarray(lblptch[ity][j].T)
      # Extract patches
      dptch = normextract(datt,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=True)
      lptch = normextract(lblt,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=False)
      sptch = np.sum(dptch,axis=1)
      if(agc): gptch = agc(sptch.astype('float32'))
      else: gptch = sptch
      # Append to output lists
      ofoc.append(dptch); olbl.append(lptch); oseg.append(gptch)
    #print("Finished examples %f"%(time.time() - beg))
    # Return patched data and labels
    foc = np.concatenate(ofoc,axis=0)
    seg = np.concatenate(oseg,axis=0)
    lbl = np.concatenate(olbl,axis=0)

    # Write the training data to HDF5 file
    wh5seg[ity].write_examples(seg,lbl)

    foclbl = np.zeros([foc.shape[0],1],dtype='float32') + focval[ity]
    wh5foc[ity].write_examples(foc,foclbl)

  # Increment position in file
  k += nw

