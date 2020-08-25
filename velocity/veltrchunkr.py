"""
Chunks velocity model creation parameters
for distribution across multiple machines

@author: Joseph Jennings
@version: 2020.08.23
"""
import numpy as np
from server.utils import splitnum

class veltrchunkr:

  def __init__(self,nchnks,nmodels,nx,ny,nz,layer,maxvel):
    """
    Creates a generator from inputs for creating
    pseudo-random velocity, reflectivity, image
    and fault models

    Parameters:
      nchnks  - length of generator (number of chunks to yield)
      nmodels - total number of models to create
      nx      - number of x-samples of velocity model
      ny      - number of y-samples of velocity model
      nz      - number of z-samples of velocity model
      layer   - width of layering for vel model
      maxvel  - maximum velocity of velocity model
    """
    # Number of chunks to create
    self.__nchnks = nchnks

    # Number of models to create
    self.__nmodels =  nmodels

    # Modeling inputs
    self.__nx = nx; self.__ny = ny; self.__nz = nz
    self.__layer = layer; self.__maxvel = maxvel

  def __iter__(self):
    """
    Defines the iterator for creating chunks

    To create the generator use gen = iter(veltrchunkr(args))
    """
    # Number of models per chunk
    modchnks = splitnum(self.__nmodels,self.__nchnks)

    ichnk = 0
    while ichnk < len(modchnks):
      # Parameters for creating the models
      mdict = {}
      mdict['nx'] = self.__nx; mdict['ny'] = self.__ny; mdict['nz'] = self.__nz
      mdict['layer'] = self.__layer; mdict['maxvel'] = self.__maxvel
      yield [modchnks[ichnk],mdict,ichnk]
      ichnk += 1

