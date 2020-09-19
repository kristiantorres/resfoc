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
hosts = ['fantastic','storm','torch','jarvis']
wph = len(hosts)*[5]
hin = create_host_list(hosts,wph)
cfile = "/homes/sep/joseph29/projects/resfoc/deeplearn/patch2dworker.py"
launch_sshworkers(cfile,hosts=hin,sleep=1,verb=1,clean=True)

taxes = sep.read_header("sigsbee_foctrimgs.H")
[nz,na,naz,nx,nm] = taxes.n
[dz,da,daz,dx,dm] = taxes.d
[oz,oa,oaz,ox,om] = taxes.o

# Size of input data for reading
nw = 10; nex = nm//nw
# Size of a single patch
ptchz = 64; ptchx = 64

# Define window
bxw = 20;  exw = nx - 20
bzw = 177; ezq = nz

context,socket = startserver()

# Create the data writing object
wh5seg = WriteToH5('/scr2/joseph29/sigsbee_fltseg1.h5',dsize=1)
wh5foc = WriteToH5('/scr2/joseph29/sigsbee_fltfoc.h5',dsize=1)

k = 0
for iex in progressbar(range(nex),"iex:"):
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
  print("Finished reading %f"%(time.time() - beg))
  beg = time.time()
  # Window the data
  focw  = foct[:,:,bxw:exw,bzw:nz]
  dfcw  = dfct[:,:,bxw:exw,bzw:nz]
  rfcw  = rfct[:,:,bxw:exw,bzw:nz]
  lblw  = lbl [:,  bxw:exw,bzw:nz]
  # Concatenate the images
  fdrptch = np.concatenate([focw,dfcw,rfcw],axis=0)
  lblptch = np.repeat(lblw,3,axis=0)
  # Create the patch chunker
  #nchnk = len(hin)
  nchnk = fdrptch.shape[0]
  pcnkr = patch2dchunkr(nchnk,fdrptch,lblptch,nzp=64,nxp=64)
  gen = iter(pcnkr)
  # Distribute and collect the results
  okeys = ['foc','seg','lbl']
  output = dstr_collect(okeys,nchnk,gen,socket)
  print("Finished examples %f"%(time.time() - beg))
  ofoc = np.concatenate(output['foc'])
  oseg = np.concatenate(outout['seg'])
  olbl = np.concatenate(output['lbl'])

  # Write the training data to HDF5 file
  #wh5seg.write_examples(oseg,olbl)

  #dummy = np.zeros([ofoc.shape[0],1],dtype='float32') - 1
  #wh5foc.write_examples(ofoc,dummy)

  # Increment position in file
  k += nw

# Clean up clients
kill_sshworkers(cfile,hosts,verb=False)

