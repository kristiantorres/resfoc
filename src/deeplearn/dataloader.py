# Classes/functions for loading in data for deep learning
from tensorflow.keras.utils import Sequence
import h5py
import numpy as np

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

