"""
Chunks Hale/BEI velocity model creation parameters
for distribution across multiple machines

@author: Joseph Jennings
@version: 2020.10.25
"""
import numpy as np
from server.utils import splitnum

class hale_veltrchunkr:

  def __init__(self,nchnks,nmodels,vzin,cano=True):
    """
    Creates a generator from inputs for creating
    pseudo-random velocity, reflectivity, image
    and fault models

    Parameters:
      nchnks  - length of generator (number of chunks to yield)
      nmodels - total number of models to create
      vzin    - input v(z) function to guide velocity model creation
      cano    - option to create a smoothly varying velocity anomaly [True]
    """
    # Number of chunks to create
    self.__nchnks = nchnks

    # Fix nz and nx for now
    nz,nx = 900,800

    # Number of models to create
    self.__nmodels =  nmodels

    # Modeling inputs
    self.__vzin = vzin

    # Flag for creating an anomaly
    self.__cano = cano

    # Anomaly creation options
    self.__nptbs   = 4;                 self.__romin  = 0.93; self.__romax = 1.00
    self.__minnaz  = 100;               self.__maxnaz = 150
    self.__minnax  = 100;               self.__maxnax = 200
    self.__mincz   = int(0.13*nz);      self.__maxcz  = int(0.22*nz)
    self.__mincx   = int(0.25*600)+120; self.__maxcx  = int(0.75*600)+120
    self.__mindist = 50
    self.__nptsz   = 2;                 self.__nptsx  = 2
    self.__octaves = 2;                 self.__period = 80
    self.__persist = 0.2
    self.__sigma   = 35

  def set_ano_pars(self,nptbs=4,romin=0.93,romax=1.00,
                   minnaz=100,maxnaz=150,minnax=100,maxnax=200,
                   mincz=None,maxcz=None,mincx=None,maxcx=None,
                   mindist=50,nptsz=2,nptsx=2,octaves=2,period=80,persist=0.2,sigma=35):
    """
    Set parameters for creating a velocity anomaly. All distance
    parameters are in units of gridpoints

    Parameters:
      nptbs   - number of anomalies to create [3]
      romin   - minimum percent value of anomaly [0.95]
      romax   - maximum percent value of anomaly [1.05]
      minnaz  - minimum size of anomaly in z [100]
      maxnaz  - maximum size of anomaly in z [150]
      minnax  - minimum size of anomaly in x [50]
      maxnax  - maximum size of anomaly in x [150]
      mincz   - minimum z bound of center of anomaly [13% of nz]
      maxcz   - maximum z bound of center of anomaly [22% of nz]
      mincx   - minimum x bound of center of anomaly [25% of nx]
      maxcx   - maximum x bound of center of anomaly [75% of nx]
      mindist - minimum distance between anomalies [50]
      nptsz   - controls the variation of the anomaly in z [2]
      nptsx   - controls the variation of the anomaly in x [2]
      octaves - number of perlin noise octaves [2]
      persist - persist of perlin noise [0.2]
      sigma   - extent of gaussian smoothing [35]
    """
    # Compute default bounds
    if(mincz is None):
      mincz = self.__mincz
    if(maxcz is None):
      maxcz = self.__maxcz
    if(mincx is None):
      mincx = self.__mincx
    if(maxcx is None):
      maxcx = self.__maxcx
    # Set parameters
    self.__nptbs   = nptbs;    self.__romin  = romin; self.__romax = romax
    self.__minnaz  = minnaz;   self.__maxnaz = maxnaz
    self.__minnax  = minnax;   self.__maxnax = maxnax
    self.__mincz   = mincz;    self.__maxcz  = maxcz
    self.__mincx   = mincx;    self.__maxcx  = maxcx
    self.__mindist = mindist
    self.__nptsz   = nptsz;    self.__nptsx  = nptsx
    self.__octaves = octaves;  self.__period = 80
    self.__persist = persist
    self.__sigma   = sigma

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
      mdict['vzin'] = self.__vzin
      # Parameters for creating the anomaly
      cdict = {}
      cdict['nptbs']   = self.__nptbs;   cdict['romin']   = self.__romin;  cdict['romax'] = self.__romax
      cdict['minnaz']  = self.__minnaz;  cdict['maxnaz']  = self.__maxnaz
      cdict['minnax']  = self.__minnax;  cdict['maxnax']  = self.__maxnax
      cdict['mincz']   = self.__mincz;   cdict['maxcz']   = self.__maxcz
      cdict['mincx']   = self.__mincx;   cdict['maxcx']   = self.__maxcx
      cdict['mindist'] = self.__mindist
      cdict['nptsz']   = self.__nptsz;   cdict['nptsx']   = self.__nptsx
      cdict['octaves'] = self.__octaves; cdict['persist'] = self.__persist
      cdict['sigma']   = self.__sigma
      yield [modchnks[ichnk],mdict,self.__cano,cdict,ichnk]
      ichnk += 1

