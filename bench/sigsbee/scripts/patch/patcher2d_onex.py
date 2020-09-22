import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from deeplearn.utils import normextract, resample, plotseglabel
from deeplearn.patch2dchunkr import patch2dchunkr
from deeplearn.dataloader import WriteToH5
from client.sshworkers import create_host_list, launch_sshworkers, kill_sshworkers
from server.utils import startserver
from server.distribute import dstr_collect
from genutils.ptyprint import progressbar, create_inttag
from deeplearn.utils import plotseglabel
import time

# IO
sep = seppy.sep()

# Start workers
hosts = ['torch','fantastic','storm','jarvis']
wph = len(hosts)*[2]
hin = create_host_list(hosts,wph)
cfile = "/homes/sep/joseph29/projects/resfoc/deeplearn/patch2dworker.py"
launch_sshworkers(cfile,hosts=hin,sleep=1,verb=1,clean=True)

taxes = sep.read_header("sigsbee_foctrimgs.H")
[nz,na,naz,nx,nm] = taxes.n
[dz,da,daz,dx,dm] = taxes.d
[oz,oa,oaz,ox,om] = taxes.o

# Size of input data for reading
#nw = 10; nex = nm//nw
nw = 20; nex = 1
# Size of a single patch
ptchz = 64; ptchx = 64

# Define window
bxw = 50;  exw = nx - 50
bzw = 177; ezq = nz

ctx,socket = startserver()

k = 0
for iex in range(nex):
  print("Batch: %s/%d"%(create_inttag(iex,nex),nex))
  # Read in the images
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
  # Window the data
  focw  = foct[:,:,bxw:exw,bzw:nz]
  dfcw  = dfct[:,:,bxw:exw,bzw:nz]
  rfcw  = rfct[:,:,bxw:exw,bzw:nz]
  lblw  = lbl [:,  bxw:exw,bzw:nz]
  # Concatenate the images
  fdrptch = [focw,dfcw,rfcw]
  lblptch = [lblw,lblw,np.zeros(lblw.shape,dtype='float32')]
  beg = time.time()
  for ity in progressbar(range(len(fdrptch)),"nty:"):
    # Create the patch chunker
    nchnk = focw.shape[0]//4
    pcnkr = patch2dchunkr(nchnk,fdrptch[ity],lblptch[ity],nzp=64,nxp=64)
    gen = iter(pcnkr)
    # Distribute and collect the results
    okeys = ['foc','seg','lbl']
    output = dstr_collect(okeys,nchnk,gen,socket,zlevel=0)

  #ntot += foc.shape[0]
  print("%f seconds"%(time.time()-beg))

  # Increment position in file
  k += nw

# Clean up clients
kill_sshworkers(cfile,hosts,verb=False)

