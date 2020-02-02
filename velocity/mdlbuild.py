import numpy as np
import velocity.evntcre8 as evntcre8
from utils.pprint import progressbar
import matplotlib.pyplot as plt

class mdlbuild:
  """ 
  Builds random geologically feasible velocity models.
  Based on the syntheticModel code from Bob Clapp

  @author: Joseph Jennings
  @version: 2020.01.22
  """

  def __init__(self,nx,dx,ny,dy,dz,nbase=50,basevel=4000):
    # Get dimensions
    self.__nbase = nbase; self.__nx = nx; self.__ny = ny
    # Get samplings
    self.__dz = dz; self.__dx = dx; self.__dy = dy
    # Create the basement layer
    self.vel = np.zeros([self.__ny,self.__nx,nbase],dtype='float32')
    self.lyr = np.zeros([self.__ny,self.__nx,nbase],dtype='int32')
    self.lbl = np.zeros([self.__ny,self.__nx,nbase],dtype='float32')
    self.vel[:] = basevel; self.lyr[:] = 0
    self.enum = 1 # Geological event number
    # Event creating object
    self.ec8 = evntcre8.evntcre8(self.__nx,self.__ny,self.__dx,self.__dy,self.__dz)

  def deposit(self,velval=1400,thick=30,band1=0.4,band2=0.02,band3=0.02,dev_pos=0.0,
              layer=23,layer_rand=0.3,dev_layer=0.26):
    """
    Creates a deposition event in the geological model. 
    
    Parameters:
      A general summary of the parameters. 
      
      Lateral variation:
      The band1-3 parameters basically are three parameters used
      to define a 3D bandpass filter which is applied to a cube of random numbers. The different
      bands allow for different amounts of variation in each direction. These and dev_pos control
      the variation of the deposit in all dimensions. Setting dev_pos to 0 will result in no
      lateral variation and the code will run much faster

      Vertical variation:
      The parameters layer,layer_rand and dev_layer dertermine the variation of the layering
      within a unit. Thick and dev_layer have the most impact where layer_rand is more of a 
      fine tuning parameter
      
      velval     - base value of velocity in the layer in m/s [1400]
      thick      - thickness of the layer in samples [30]
      band1      - bandpass parameter for axis 1 [0.4]
      band2      - bandpass parameter for axis 2 [0.02]
      band3      - bandpass parameter for axis 3 [0.02]
      dev_pos    - Determines the strength of the varation of the velocity in all directions by
                   bandpassing randmom numbers (setting to 0 will make it much faster) [0.0]
      layer      - Changes the thickness of the thin layering (bigger number is thicker) [23]
      layer_rand - Also change the thickness of the thin layering [0.0]. A larger number leads
                   to more thin bedding within a layer
      dev_layer  - Changes the variation of the velocity within a layer. Larger values lead to more variation
                   Set to 0 will result in homogeonous layer [0.26]
    """
    # First, expand the model
    nzin = self.vel.shape[2]
    nzot = int(nzin + thick)
    velot = np.zeros([self.__ny,self.__nx,nzot],dtype='float32')
    lyrot = np.zeros([self.__ny,self.__nx,nzot],dtype='int32')
    self.ec8.expand(thick,0,nzin,self.lyr,self.vel,nzot,lyrot,velot)
    # Next create the deposit
    self.ec8.deposit(velval,
                     band1,band2,band3,
                     layer,layer_rand,dev_layer,dev_pos,
                     nzot,lyrot,velot)
    # Update the layer and velocity models
    self.vel = velot
    self.lyr = lyrot

  def fault(self,begx=0.5,begy=0.5,begz=0.5,daz=8000,dz=7000,azim=180,
      theta_die=12,theta_shift=4.0,dist_die=0.3,perp_die=0.5,dirf=0.1,thresh=50):
    """
    Creates a fault event in the geologic model

    Parameters:
      azim        - Azimuth of fault [180]
      begx        - Relative location of the beginning of the fault in x [0.5]
      begy        - Relative location of the beginning of the fault in y [0.5]
      begz        - Relative location of the beginning of the fault in z [0.5]
      dz          - Distance away from the center of a circle in z [7000]
      daz         - Distance away in azimuth [8000]
      perp_die    - Controls the die off perpendicular to the fault
                    (e.g., if fault is visible in x, die off is in y).
                    Large number results in slower dieoff [0.5]
      dist_die    - Controls the die off in the same plane of the fault 
                    (e.g., if fault is visible in x, die off is also in x)
                    Large number results in slower dieoff [0.3]
      theta_die   - Controls the die off along the fault (essentially the throw). The larger 
                    the number the larger the fault will be. Acts similar to daz. [12]
      theta_shift - Shift in theta for fault [4.0]
      dirf        - Direction of fault movement [0.1]
    """
    # Create the output velocity, layermodel and label
    nz = self.vel.shape[2]
    velot = np.zeros(self.vel.shape,dtype='float32')
    lyrot = np.zeros(self.lyr.shape,dtype='int32')
    lbltp = np.zeros(self.vel.shape,dtype='float32')
    lblot = np.zeros(self.vel.shape,dtype='float32')
    # Create the fault
    self.ec8.fault(nz,self.lyr,self.vel,
                   azim,begx,begy,begz,dz,daz,
                   theta_shift,perp_die,dist_die,theta_die,dirf,
                   lyrot,velot,lbltp)
    # Update layer and velocity models
    self.vel = velot
    self.lyr = lyrot
    # Compute z-derivative of label
    self.ec8.laplacian(nz,lbltp,lblot)
    # Apply a threshold
    idx = np.abs(lblot) > thresh
    lblot[ idx] = 1; lblot[~idx] = 0
    # Update label
    if(self.lbl.shape[2] != lblot.shape[2]):
      ndiff = lblot.shape[2] - self.lbl.shape[2]
      self.lbl = np.pad(self.lbl,((0,0),(0,0),(ndiff,0)),'constant')
      self.lbl += lblot
    else:
      self.lbl += lblot
    # Apply final threshold to label
    idx = self.lbl > 1
    self.lbl[idx] = 1

  def get_label(self):
    """
    Gets the fault labels and ensures the label is the same size
    as the current velocity model size
    """
    nzv = self.vel.shape[2]
    nzl = self.lbl.shape[2]
    if(nzv == nzl):
      return self.lbl
    else:
      ndiff = nzv - nzl
      return np.pad(self.lbl,((0,0),(0,0),(ndiff,0)),'constant')

  def smallfault_block(self,nfault=5,azim=0.0,begz=0.3,begx=0.3,begy=0.3,xdir=True):
    """
    Puts in a small fault block system. For now, only will give nice faults along
    0,90,180,270 azimuths

    Parameters:
      nfault - number of faults in the system [5]
      azim   - azimuth along which faults are oriented [0.0]
      begz   - beginning position in z for fault (same for all) [0.3]
      begx   - beginning position in x for system [0.5]
      begy   - beginning position in y for system [0.5]
      xdir   - move along the x direction [True]. If False, will move the fault
               system in the y direction (all faults will still have same azimuth)
    """
    #TODO: might be a good idea to have small variation in z
    dx = 0.0; dy = 0.0
    signx = 1; signy = 1
    if(xdir):
      dx = 0.1
    else:
      dy = 0.1
    if(begx > 0.5):
      signx = -1
    if(begy > 0.5):
      signy = -1
    for ifl in progressbar(range(nfault), "nfaults:", 40):
      self.fault(begx=begx,begy=begy,begz=begz,daz=8000,dz=5000,azim=azim,theta_die=11.0,theta_shift=4.0,dist_die=0.3,perp_die=1.0)
      # Move along x or y
      begx += signx*dx; begy += signy*dy

  def largefault_block(nfault,azim,begz):
    pass

  def sliding_block(nfault,azim,begz):
    pass

  def graben_block(begz,azim):
    pass

  def horstgraben_block(begz,azim):
    pass

  def smallfault(azim=0.0,begz=0.3):
    pass

  def largefault(azim=0.0,begz=0.5):
    pass

  def slidingfault(azim=0.0,begz=0.5):
    pass

  def fold():
    pass
