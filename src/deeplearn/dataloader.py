# Classes/functions for loading in data for deep learning
from tensorflow.keras.utils import Sequence
import h5py
import numpy as np
import random
import subprocess

class resmig_generator_h5(Sequence):

  def __init__(self, fname, batch_size):
    self.hfin = h5py.File(fname,'r')
    self.hfkeys = list(self.hfin.keys())
    self.nb = int(len(self.hfkeys)/2) # Half is features, half is labels
    self.indices = np.arange(self.nb)
    self.batch_size = batch_size

  def get_xyshapes(self):
    return self.hfin[self.hfkeys[0]].shape[1:], self.hfin[self.hfkeys[self.nb]].shape[1:]

  def __len__(self):
    return self.nb

  def __getitem__(self,idx):
    xb = self.hfin[self.hfkeys[idx]]
    yb = np.expand_dims(self.hfin[self.hfkeys[idx + self.nb]],axis=-1)
    return xb, yb

#

def splith5(fin,f1,f2,split=0.8,rand=False,clean=True):
  """
  Splits an H5 file into two other files.
  Useful for splitting data into training and validation
  """
  hfin = h5py.File(fin,'r')
  hf1  = h5py.File(f1,'w')
  hf2  = h5py.File(f2,'w')
  keys = list(hfin.keys())
  nb   = int(len(keys)/2)
  nf1 = int(split*nb)
  nf2 = nb - nf1
  if(rand):
    choices = list(range(nb))
    idxs = random.sample(choices,nf1)
  else:
    idxs = list(range(nf1))
  for idx in range(nb):
    if idx in idxs:
      hfin.copy(keys[idx],hf1)
      hfin.copy(keys[idx+nb],hf1)
    else:
      hfin.copy(keys[idx],hf2)
      hfin.copy(keys[idx+nb],hf2)
  hfin.close()
  hf1.close()
  hf2.close()
  # Remove the original file
  if(clean):
    sp = subprocess.check_call('rm %s'%(fin),shell=True)

def load_alldata(trfile,vafile,dsize):
  """ Loads all data and labels into numpy arrays """
  # Get training number of examples
  hftr = h5py.File(trfile,'r')
  trkeys = list(hftr.keys())
  ntr = int(len(trkeys)/2)
  # Get the validation number of examples
  if(vafile != None):
    hfva = h5py.File(vafile,'r')
    vakeys = list(hfva.keys())
    nva = int(len(vakeys)/2)
  else:
    nva = 0; vakeys = []
  # Get shape of examples
  xshape = hftr[trkeys[0]].shape
  yshape = hftr[trkeys[0+ntr]].shape
  # Allocate output arrays
  if(len(xshape) == 4):
    allx = np.zeros([(ntr+nva)*dsize,xshape[1],xshape[2],xshape[3]],dtype='float32')
  elif(len(xshape) == 3):
    allx = np.zeros([(ntr+nva)*dsize,xshape[1],xshape[2]],dtype='float32')
  ally = np.zeros([(ntr+nva)*dsize,yshape[1],yshape[2],1],dtype='float32')
  k = 0
  # Get all training examples
  for itr in range(ntr):
    for iex in range(dsize):
      allx[k,:,:,:]  = hftr[trkeys[itr]    ][iex,:,:,:]
      ally[k,:,:,0]  = hftr[trkeys[itr+ntr]][iex,:,:]
      k += 1
  # Get all validation examples
  for iva in range(nva):
    for iex in range(dsize):
      allx[k,:,:,:]  = hfva[vakeys[iva]    ][iex,:,:,:]
      ally[k,:,:,0]  = hfva[vakeys[iva+nva]][iex,:,:]
      k += 1
  # Close the files
  hftr.close()
  if(vafile != None): hfva.close()

  return allx,ally
