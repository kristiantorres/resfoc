import torch
from torch.utils.data import Dataset
import h5py

class SigsbeeFocData(Dataset):
  """ Sigsbee focusing data """

  def __init__(self,h5file,nexmax=None):
    """
    Creates a sigsbee focusing dataset

    Parameters:
      h5file - Path to the H5 file containing the training data
      nexmax - maximum number of examples to use during training [None]
    """
    self.__hfin = h5py.File(h5file,'r')
    self.__keys = list(self.__hfin.keys())
    self.__nex  = len(self.__keys)//2
    if(nexmax is not None):
      if(nexmax <= self.__nex):
        self.__nexmax = nexmax
      else:
        self.__nexmax = self.__nex
    else:
      self.__nexmax = self.__nex

  def __len__(self):
    return self.__nexmax

  def __getitem__(self,idx):

    image = self.__hfin[self.__keys[idx           ]]
    label = self.__hfin[self.__keys[idx+self.__nex]]

    sample = {'img': torch.from_numpy(image[:].squeeze(0)),
              'lbl': torch.from_numpy(label[:]).squeeze().unsqueeze(0)}

    return sample

  def __del__(self):
    try:
      self.__hfin.close()
    except:
      pass

