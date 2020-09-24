import torch
from torch.utils.data import Dataset
import numpy as np
from deeplearn.dataloader import load_alldata
import h5py

class SigsbeeFocDataGPU(Dataset):
  """ Sigsbee focusing data """

  def __init__(self,h5file,device,begex=None,endex=None,verb=False):
    """
    Creates a sigsbee focusing dataset.
    Requires that all of the data can fit into GPU memory

    Parameters:
      h5file - Path to the H5 file containing the training data
      device - Device to which to copy the data
    """
    # Load in all of the data
    allx,ally = load_alldata(h5file,None,1,begex,endex,verb)
    # Convert to torch tensors
    tallx = torch.from_numpy(allx)
    tally = torch.from_numpy(ally[:,0,:])
    # Copy all data to the device
    self.__gallx = tallx.to(device)
    self.__gally = tally.to(device)

    self.__nex = allx.shape[0]

  def __len__(self):
    return self.__nex

  def __getitem__(self,idx):

    sample = {'img': self.__gallx[idx],
              'lbl': self.__gally[idx]}

    return sample

  def __del__(self):
    try:
      self.__hfin.close()
    except:
      pass

