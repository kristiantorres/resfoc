"""
Chunks training data examples
for distributing patching across multiple machines
for fault segmentation

@author: Joseph Jennings
@version: 2020.09.17
"""
import numpy as np
from server.utils import splitnum

class patch2dchunkr:

  def __init__(self,nchnks,dat,lbl,nzp,nxp,strdz=None,strdx=None,
               transp=False,agc=True):
    """
    Creates a generator from inputs for creating
    patches of images and labels for fault focusing
    and segmentation

    Parameters:
      nchnks - length of generator (number of chunks to yield)
      dat    - input seismic images [nex,na,nx,nz]
      lbl    - input labels [nex,nx,nz]
      nzp    - size of the patch in z direction
      nxp    - size of the patch in x direction
      strdz  - patch stride in z direction [nzp/2]
      strdx  - patch stride in x direction [nxp/2]
      transp - inputs have shape [nex,nz,nx]
      agc    - apply agc to input images [True]
    """
    # Number of chunks to create
    self.__nchnks = nchnks

    # Check the dimensions of the inputs
    if(dat.shape[2:] != lbl.shape[1:]):
      raise Exception("Labels and data must have same spatial dimensions")

    # Save the inputs
    self.__dat = dat; self.__lbl = lbl

    # Get dimensions of data
    self.__transp = transp
    if(self.__transp):
      [self.__nex,self.__na,self.__nz,self.__nx] = self.__dat.shape
    else:
      [self.__nex,self.__na,self.__nx,self.__nz] = self.__dat.shape

    # Patching inputs
    self.__nzp   = nzp;   self.__nxp   = nxp
    self.__strdz = strdz; self.__strdx = strdx
    if(self.__strdz is None): self.__strdz = int(self.__nzp/2 + 0.5)
    if(self.__strdx is None): self.__strdx = int(self.__nxp/2 + 0.5)

    # AGC flag
    self.__agc = agc

  def __iter__(self):
    """
    Defines the iterator for creating chunks

    To create the generator use gen = iter(fltsegpatchchunkr(args))
    """
    # Number of examples per chunk
    pchcnks = splitnum(self.__nex,self.__nchnks)

    ichnk = 0; begex = 0; endex = 0
    while ichnk < len(pchcnks):
      # Parameters for creating the models
      pdict = {}
      # Chunk the data and labels
      begex = endex; endex += pchcnks[ichnk]
      pdict['dat'] = self.__dat[begex:endex]; pdict['lbl'] = self.__lbl[begex:endex]
      # Chunk the patching paramters
      pdict['nzp']    = self.__nzp;    pdict['nxp']   = self.__nxp
      pdict['strdz']  = self.__strdz;  pdict['strdx'] = self.__strdx
      # Additonal parameters
      adict = {}
      adict['transp'] = self.__transp; adict['agc']   = self.__agc
      yield [pdict,adict,ichnk]
      ichnk += 1

