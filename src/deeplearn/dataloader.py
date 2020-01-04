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
    yb = self.hfin[self.hfkeys[idx + self.nb]]
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

