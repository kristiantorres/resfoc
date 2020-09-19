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

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm','torch','jarvis']
wph = len(hosts)*[10]
hin = create_host_list(hosts,wph)
cfile = "/homes/sep/joseph29/projects/resfoc/deeplearn/fltsegpatchworker.py"
launch_sshworkers(cfile,hosts=hin,sleep=1,verb=1,clean=True)

taxes = sep.read_header("sigsbee_foctrimgs.H")
[nz,na,naz,nx,nm] = taxes.n
[dz,da,daz,dx,dm] = taxes.d
[oz,oa,oaz,ox,om] = taxes.o

# Size of input data for reading
nw = 20; nex = nm//nw
# Size of a single patch
ptchz = 64; ptchx = 64

# Define window
bxw = 20;  exw = nx - 20
bzw = 177; ezq = nz

context,socket = startserver()

# Create the data writing object
wh5 = WriteToH5('/scr2/joseph29/sigsbee_focdefres_sort.h5',dsize=1)

k = 0
for iex in progressbar(range(nex),"iex:"):
  # Read in the images
  beg = time.time()
  faxes,foc = sep.read_wind("sigsbee_foctrimgs.H",fw=k,nw=nw)
  foc   = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
  foct  = np.ascontiguousarray(np.transpose(foc[:,:,0,:,:],(0,2,1,3)))
  focts = np.sum(foct,axis=1)
  # Read in defocused images
  daxes,dfc = sep.read_wind("sigsbee_deftrimgs.H",fw=k,nw=nw)
  dfc   = np.ascontiguousarray(dfc.reshape(daxes.n,order='F').T).astype('float32')
  dfct  = np.ascontiguousarray(np.transpose(dfc[:,:,0,:,:],(0,2,1,3)))
  dfcts = np.sum(dfct,axis=1)
  # Read in residually focused images
  raxes,rfc = sep.read_wind("sigsbee_restrimgs.H",fw=k,nw=nw)
  rfc   = np.ascontiguousarray(rfc.reshape(daxes.n,order='F').T).astype('float32')
  rfct  = np.ascontiguousarray(np.transpose(rfc[:,:,0,:,:],(0,2,1,3)))
  rfcts = np.sum(rfct,axis=1)
  print("Finished reading %f"%(time.time() - beg))
  beg = time.time()
  # Window the data
  focw  = focts[:,bxw:exw,bzw:nz]
  dfcw  = dfcts[:,bxw:exw,bzw:nz]
  rfcw  = rfcts[:,bxw:exw,bzw:nz]
  # Concatenate the images
  fdrptch = np.concatenate([focw,dfcw,rfcw],axis=0)
  labels  = np.zeros(fdrptch.shape,dtype='float32') + -1
  # Create the patch chunker
  nchnk = len(hin)
  pcnkr = fltsegpatchchunkr(nchnk,fdrptch,labels,nzp=64,nxp=64)
  gen = iter(pcnkr)
  # Distribute and collect the results
  okeys = ['dat','lbl','cid']
  output = dstr_collect(okeys,nchnk,gen,socket)
  print("Finished examples %f"%(time.time() - beg))
  # Sort so they are in the correct order
  idx  = np.argsort(output['cid'])
  odat = np.concatenate(np.asarray(output['dat'])[idx],axis=0)
  olbl = np.concatenate(np.asarray(output['lbl'])[idx],axis=0)

  # Write the training data to HDF5 file
  wh5.write_examples(odat,olbl)

  # Increment position in file
  k += nw

# Clean up clients
kill_sshworkers(cfile,hosts,verb=False)

