"""
Chunks indices and file information for patching
training data across a cluster

@author: Joseph Jennings
@version: 2020.09.20
"""
import os
import numpy as np
from server.utils import splitnum

class patch2dchunkr:

  def __init__(self,nchnks,nex,ffile,dfile,rfile,lfile,
               nzp,nxp,strdz=None,strdx=None,
               bxw=0,exw=None,bzw=0,ezw=None,nwmax=20,
               transp=False,agc=True,logdir=None,verb=False):
    """
    Creates a generator from inputs for creating
    patches of images and labels for fault focusing
    and segmentation

    Parameters:
      nchnks - length of generator (number of chunks to yield)
      nex    - total number of examples to patch
      ffile  - file path to focused images
      dfile  - file path to defocused images
      rfile  - file path to residually defocused images
      lfile  - file path to fault labels
      nzp    - size of the patch in z direction
      nxp    - size of the patch in x direction
      strdz  - patch stride in z direction [nzp/2]
      strdx  - patch stride in x direction [nxp/2]
      bxw    - first sample to window in x [0]
      exw    - last sample to window in x  [nx]
      bzw    - first sample to window in z [0]
      ezw    - last sample to window in z [nz]
      nwmax  - number of examples to read in at once [20]
      agc    - apply agc to input images [True]
      transp - inputs have shape [nex,nz,nx]
      verb   - verbosity flag [False]
    """
    # Number of chunks to create
    self.__nchnks = nchnks

    # Total number of examples to patch
    self.__nex = nex

    # Maximum number of examples per read
    self.__nwmax = nwmax

    # Make sure files exist
    if(not os.path.exists(ffile)):
      raise Exception("File %s does not exist"%(ffile))
    else:
      self.__ffile = ffile
    if(not os.path.exists(dfile)):
      raise Exception("File %s does not exist"%(dfile))
    else:
      self.__dfile = dfile
    if(not os.path.exists(rfile)):
      raise Exception("File %s does not exist"%(rfile))
    else:
      self.__rfile = rfile
    if(not os.path.exists(lfile)):
      raise Exception("File %s does not exist"%(lfile))
    else:
      self.__lfile = lfile

    # Patching inputs
    self.__bxw   = bxw;   self.__exw   = exw
    self.__bzw   = bzw;   self.__ezw   = ezw
    self.__nzp   = nzp;   self.__nxp   = nxp
    self.__strdz = strdz; self.__strdx = strdx
    if(self.__strdz is None): self.__strdz = int(self.__nzp/2 + 0.5)
    if(self.__strdx is None): self.__strdx = int(self.__nxp/2 + 0.5)

    # AGC and transpose flags
    self.__agc, self.__transp = agc, transp

    # Logging directory
    self.__logdir = logdir
    if(self.__logdir is None):
      self.__logdir = '/homes/sep/joseph29/'

    self.__verb = verb

  def set_window_pars(self,bxw,exw,bzw,ezw) -> None:
    """
    Overrides default parameters set in the constructor for the patching
    parameters

    Parameters:
      bxw - first sample to window in x [0]
      exw - last sample to window in x [nx]
      bzw - first sample to window in z [0]
      ezw - last sample to window in z [nz]
    """
    self.__bxw = bxw; self.__exw = exw
    self.__bzw = bzw; self.__ezw = ezw

  def __iter__(self):
    """
    Defines the iterator for creating chunks

    To create the generator use gen = iter(patch2dchunkr(args))
    """
    # Number of examples per chunk
    pchcnks = splitnum(self.__nex,self.__nchnks)

    # Number of examples to read in per file
    nws = []
    for num in pchcnks:
      divs = self.divisors(num)
      nws.append(divs[np.argmin(self.remove_nonzero(self.__nwmax - divs))])

    if(self.__verb):
      print("nex per worker: ",*pchcnks)
      print("nw  per worker: ",*nws)

    ichnk = 0; begpos = 0; endpos = 0
    while ichnk < len(pchcnks):
      # File reading positions
      begpos = endpos; endpos += pchcnks[ichnk]
      # Parameters for reading the files
      rdict = {}
      rdict['ffile'] = self.__ffile;  rdict['dfile'] = self.__dfile
      rdict['rfile'] = self.__rfile;  rdict['lfile'] = self.__lfile
      rdict['fw']    = begpos;        rdict['nw']    = nws[ichnk]
      rdict['nm']    = pchcnks[ichnk]
      # Patching parameters
      pdict = {}
      pdict['bxw']    = self.__bxw;    pdict['exw']   = self.__exw
      pdict['bzw']    = self.__bzw;    pdict['ezw']   = self.__ezw
      pdict['nzp']    = self.__nzp;    pdict['nxp']   = self.__nxp
      pdict['strdz']  = self.__strdz;  pdict['strdx'] = self.__strdx
      # Additonal parameters
      adict = {}
      adict['transp'] = self.__transp; adict['agc']   = self.__agc
      yield [rdict,pdict,adict,self.__logdir,ichnk]
      ichnk += 1

  def divisors(self,num) -> np.ndarray:
    """ Returns the divisors of the number """
    return np.asarray([i for i in range(1,num+1) if(num%i == 0)],dtype='int32')

  def remove_nonzero(self,arr) -> np.ndarray:
    """ Removes the non-zero elements """
    idx = arr >= 0
    return arr[idx]

