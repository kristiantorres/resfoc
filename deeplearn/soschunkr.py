"""
Chunks files for smoothing training images across
a cluster

@author: Joseph Jennings
@version: 2020.10.31
"""
import os
import numpy as np
from server.utils import splitnum

class soschunkr:

  def __init__(self,nchnks,imgfiles,verb=False):
    """
    Creates a generator from inputs for creating
    smoothing images fault focusing
    and segmentation

    Parameters:
      nchnks   - length of generator (number of chunks to yield)
      imgfiles - list of image files
      verb     - verbosity flag [False]
    """
    # Number of chunks to create
    self.__nchnks = nchnks

    # Number of examples
    self.__nex = len(imgfiles)

    # Make sure paths exist
    idir = os.path.dirname(imgfiles[0])
    if(not os.path.exists(idir)):
      raise Exception("Path to image or label dir does not exist")

    self.__imgfiles = imgfiles

    # Verbosity flag
    self.__verb = verb

  def __iter__(self):
    """
    Defines the iterator for creating chunks

    To create the generator use gen = iter(soschunkr(args))
    """
    # Number of examples per chunk
    fchcnks = splitnum(self.__nex,self.__nchnks)

    ichnk = 0; begpos = 0; endpos = 0
    while ichnk < len(fchcnks):
      # File reading positions
      begpos = endpos; endpos += fchcnks[ichnk]
      # Parameters for reading the files
      fdict = {}
      fdict['imgfiles'] = self.__imgfiles[begpos:endpos]
      yield [fdict,ichnk,self.__verb]
      ichnk += 1

