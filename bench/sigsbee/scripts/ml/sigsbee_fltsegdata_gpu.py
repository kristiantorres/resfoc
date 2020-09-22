import torch
from torch.utils.data import Dataset
import numpy as np
from deeplearn.dataloader import load_alldata
import h5py

class SigsbeeFltSegDataGPU(Dataset):
  """ Sigsbee Fault segmentation data """

  def __init__(self,h5file,device,begex=None,endex=None,verb=False):
    """
    Creates a sigsbee fault segmentation dataset.
    Requires that all of the data can fit into GPU memory

    Parameters:
      h5file - Path to the H5 file containing the training data
      device - Device to which to copy the data
    """
    # Load in all of the data
    allx,ally = load_alldata(h5file,None,1,begex,endex,verb)
    #TODO: in the future, I will remove the keras channels last thing
    allxt = np.ascontiguousarray(np.transpose(allx,(0,3,1,2)))
    allyt = np.ascontiguousarray(np.transpose(ally,(0,3,1,2)))
    # Convert to torch tensors
    tallx = torch.from_numpy(allxt)
    tally = torch.from_numpy(allyt)
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

