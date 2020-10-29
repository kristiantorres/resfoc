"""
Chunks files for patching training data across
a cluster

@author: Joseph Jennings
@version: 2020.10.27
"""
import os
import numpy as np
from server.utils import splitnum

class patch2dfltseg_chunkr:

  def __init__(self,nchnks,imgfiles,lblfiles,
               nzp,nxp,strdz=None,strdx=None,smooth=True,
               verb=False):
    """
    Creates a generator from inputs for creating
    patches of images and labels for fault focusing
    and segmentation

    Parameters:
      nchnks   - length of generator (number of chunks to yield)
      imgfiles - list of image files
      lblfiles - list of corresponding label files
      nzp      - size of the patch in z direction
      nxp      - size of the patch in x direction
      strdz    - patch stride in z direction [nzp/2]
      strdx    - patch stride in x direction [nxp/2]
      smooth   - flag for applying structure-oriented smoothing [True]
      verb     - verbosity flag [False]
    """
    # Number of chunks to create
    self.__nchnks = nchnks

    # Number of examples
    self.__nex = len(imgfiles)
    if(self.__nex != len(lblfiles)):
      raise Exception("Number of image and label files must be the same")

    # Make sure paths exist
    idir = os.path.dirname(imgfiles[0])
    ldir = os.path.dirname(lblfiles[0])
    if(not os.path.exists(idir) or not os.path.exists(ldir)):
      raise Exception("Path to image or label dir does not exist")

    self.__imgfiles = imgfiles
    self.__lblfiles = lblfiles

    # Patching inputs
    self.__nzp   = nzp;   self.__nxp   = nxp
    self.__strdz = strdz; self.__strdx = strdx
    if(self.__strdz is None): self.__strdz = int(self.__nzp/2 + 0.5)
    if(self.__strdx is None): self.__strdx = int(self.__nxp/2 + 0.5)

    # Smoothing flag
    self.__smooth = smooth

    # Verbosity flag
    self.__verb = verb

  def __iter__(self):
    """
    Defines the iterator for creating chunks

    To create the generator use gen = iter(patch2dchunkr(args))
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
      fdict['lblfiles'] = self.__lblfiles[begpos:endpos]
      # Patching parameters
      pdict = {}
      pdict['nzp']    = self.__nzp;    pdict['nxp']   = self.__nxp
      pdict['strdz']  = self.__strdz;  pdict['strdx'] = self.__strdx
      # Flags
      ldict = {}
      ldict['smooth'] = self.__smooth; ldict['verb'] = self.__verb
      yield [fdict,pdict,ichnk,ldict]
      ichnk += 1

