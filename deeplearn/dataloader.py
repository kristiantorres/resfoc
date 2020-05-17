# Classes/functions for loading in data for deep learning
from tensorflow.keras.utils import Sequence
import h5py
import numpy as np
import random
import subprocess
from utils.ptyprint import progressbar

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
  for idx in progressbar(range(nb), "nbatches:"):
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
  for itr in progressbar(range(ntr), "numtr:"):
    for iex in range(dsize):
      allx[k,:,:,:]  = hftr[trkeys[itr]    ][iex,:,:,:]
      if(len(yshape) == 3):
        ally[k,:,:,0]  = hftr[trkeys[itr+ntr]][iex,:,:]
      else:
        ally[k,:,:,0]  = hftr[trkeys[itr+ntr]][iex,:,:,0]
      k += 1
  # Get all validation examples
  for iva in progressbar(range(nva), "numva:"):
    for iex in range(dsize):
      allx[k,:,:,:]  = hfva[vakeys[iva]    ][iex,:,:,:]
      if(len(yshape) == 3):
        ally[k,:,:,0]  = hfva[vakeys[iva+nva]][iex,:,:]
      else:
        ally[k,:,:,0]  = hfva[vakeys[iva+nva]][iex,:,:,0]
      k += 1
  # Close the files
  hftr.close()
  if(vafile != None): hfva.close()

  return allx,ally

def load_allflddata(fldfile,dsize):
  """ Loads all field (unlabeled) data into a numpy array """
  # Get training number of examples
  hffld = h5py.File(fldfile,'r')
  fldkeys = list(hffld.keys())
  nfld = int(len(fldkeys)/2)
  # Get shape of examples
  xshape = hffld[fldkeys[0]].shape
  # Allocate output arrays
  if(len(xshape) == 4):
    allx = np.zeros([(nfld)*dsize,xshape[1],xshape[2],xshape[3]],dtype='float32')
  elif(len(xshape) == 3):
    allx = np.zeros([(nfld)*dsize,xshape[1],xshape[2]],dtype='float32')
  k = 0
  # Get all field examples
  for ifld in progressbar(range(nfld), "numfld:"):
    for iex in range(dsize):
      allx[k,:,:,:]  = hffld[fldkeys[ifld]][iex,:,:,:]
      k += 1

  return allx

def load_allpatchdata(trfile,vafile,dsize):
  """
  Loads all data and labels into numpy arrays

  Works both for the SSIM residual migration training
  and the patchwise fault classification training
  """
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
  ally = np.zeros([(ntr+nva)*dsize,yshape[1]],dtype='float32')
  k = 0
  # Get all training examples
  for itr in progressbar(range(ntr), "numtr:"):
    for iex in range(dsize):
      allx[k,:,:,:]  = hftr[trkeys[itr]    ][iex,:,:,:]
      ally[k,:]  = hftr[trkeys[itr+ntr]][iex,:]
      k += 1
  # Get all validation examples
  for iva in progressbar(range(nva), "numva:"):
    for iex in range(dsize):
      allx[k,:,:,:]  = hfva[vakeys[iva]    ][iex,:,:,:]
      ally[k,:]  = hfva[vakeys[iva+nva]][iex,:]
      k += 1
  # Close the files
  hftr.close()
  if(vafile != None): hfva.close()

  return allx, ally

def load_allssimcleandata(trfile,vafile):
  """ Loads a cleaned and flattened ssim data into numpy arrays """
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
  allx = np.zeros([(ntr+nva),xshape[0],xshape[1],xshape[2]],dtype='float32')
  ally = np.zeros([(ntr+nva),yshape[0]],dtype='float32')
  # Get all training examples
  for itr in progressbar(range(ntr), "numtr:"):
    allx[itr,:,:,:]  = hftr[trkeys[itr]    ][:]
    ally[itr,:]      = hftr[trkeys[itr+ntr]][:]
  # Get all validation examples
  for iva in progressbar(range(nva), "numva:"):
    allx[iva,:,:,:]  = hfva[vakeys[iva]    ][:]
    ally[iva,:]      = hfva[vakeys[iva+nva]][:]
  # Close the files
  hftr.close()
  if(vafile != None): hfva.close()

  return allx, ally

