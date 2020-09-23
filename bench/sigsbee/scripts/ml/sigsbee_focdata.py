import torch
from torch.utils.data import Dataset
import h5py

class SigsbeeFocData(Dataset):
  """ Sigsbee focusing data """

  def __init__(self,h5file):
    """
    Creates a sigsbee focusing dataset

    Parameters:
      h5file - Path to the H5 file containing the training data
    """
    self.__hfin = h5py.File(h5file,'r')
    self.__keys = list(self.__hfin.keys())
    self.__nex  = len(self.__keys)//2

  def __len__(self):
    return self.__nex

  def __getitem__(self,idx):

    image = self.__hfin[self.__keys[idx           ]]
    label = self.__hfin[self.__keys[idx+self.__nex]]

    sample = {'img': torch.from_numpy(image[:,:,:,:,0]),
              'lbl': torch.from_numpy(label[:])}

    return sample

  def __del__(self):
    try:
      self.__hfin.close()
    except:
      pass

