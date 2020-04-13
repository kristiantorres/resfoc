import h5py
import glob
import numpy as np
from utils.ptyprint import progressbar, create_inttag

dpath = '/scr2/joseph29/data/train/seis/'
lpath = '/scr2/joseph29/data/train/fault/'

# Get all of the files in the directories
dfiles = sorted(glob.glob(dpath+'*.dat'))
lfiles = sorted(glob.glob(lpath+'*.dat'))

nfiles = len(dfiles)

ntot = nfiles*128
dattot = np.zeros([ntot,128,128])
lbltot = np.zeros([ntot,128,128])

beg = 0; end = 128
for ifile in progressbar(range(nfiles), "nfiles:"):
  # Read in the files
  with open(dfiles[ifile],'r') as f:
    dat = np.fromfile(f,dtype='<f')
  with open(lfiles[ifile],'r') as f:
    lbl = np.fromfile(f,dtype='<f')
  # Reshape them to 3D
  dat = dat.reshape([128,128,128])
  lbl = lbl.reshape([128,128,128])
  # Append them to the output array
  dattot[beg:end,:,:] = dat[:]
  lbltot[beg:end,:,:] = lbl[:]
  # Update the intervals
  beg += 128; end += 128

bsize = 20
nbs = int(ntot/bsize)

datbch = np.zeros([bsize,128,128])
lblbch = np.zeros([bsize,128,128])
hf = h5py.File('/scr2/joseph29/data/train/mine.h5','w')
# Split them into batches and save to an H5 file
ctr = 0
for ib in progressbar(range(nbs),"nbatches:"):
  datatag = create_inttag(ib,nbs)
  for iex in range(bsize):
    datbch[iex,:,:] = dattot[ctr,:,:]
    lblbch[iex,:,:] = lbltot[ctr,:,:]
    # Update total counter
    ctr += 1
  # Write to H5 file
  hf.create_dataset("x"+datatag, (bsize,128,128,1), data=np.expand_dims(datbch,axis=-1), dtype=np.float32)
  hf.create_dataset("y"+datatag, (bsize,128,128,1), data=np.expand_dims(lblbch,axis=-1), dtype=np.float32)

hf.close()
