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

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm','torch','jarvis']
wph = len(hosts)*[5]
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
wh5 = WriteToH5('/net/fantastic/scr2/joseph29/sigsbee_fltseg_dstr.h5',dsize=1)

k = 0
for iex in progressbar(range(nex),"iex:"):
  # Read in the images
  faxes,foc = sep.read_wind("sigsbee_foctrimgs.H",fw=k,nw=nw)
  foc  = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
  foct = np.ascontiguousarray(np.transpose(foc[:,:,0,:,:],(0,2,1,3)))
  focts = np.sum(foct,axis=1)
  # Read in the labels
  laxes,lbl = sep.read_wind("sigsbee_trlblsint.H",fw=k,nw=nw)
  lbl = np.ascontiguousarray(lbl.reshape(laxes.n,order='F').T).astype('float32')
  # Window the data
  foctsw = focts[:,bxw:exw,bzw:nz]
  lblw   = lbl  [:,bxw:exw,bzw:nz]
  # Create the patch chunker
  nchnk = len(hin)
  pcnkr = fltsegpatchchunkr(nchnk,foctsw,lblw,nzp=64,nxp=64)
  gen = iter(pcnkr)

  # Distribute and collect the results
  okeys = ['dat','lbl']
  output = dstr_collect(okeys,nchnk,gen,socket)
  odat = np.concatenate(output['dat'])
  olbl = np.concatenate(output['lbl'])

  # Write the training data to HDF5 file
  wh5.write_examples(odat,olbl)

  # Increment position in file
  k += nw

# Clean up clients
kill_sshworkers(cfile,hosts,verb=False)

