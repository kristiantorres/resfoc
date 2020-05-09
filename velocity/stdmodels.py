"""
Functions that create "standard" velocity models
Current models are:
  1. Layered v(z) with three faults
  2. Heavily faulted models with complex velocity

@author: Joseph Jennings
@version: 2020.05.01
"""
import numpy as np
import velocity.mdlbuild as mdlbuild
from scaas.wavelet import ricker
from utils.ptyprint import progressbar, create_inttag
import utils.rand as rndut
import deeplearn.utils as dlut
from scipy.ndimage import gaussian_filter
from utils.signal import bandpass

def layeredfaults2d(nz=512,nx=1000,dz=12.5,dx=25.0,nlayer=21,minvel=1600,maxvel=3000,rect=0.5,
                    nfx=3,ofx=0.4,dfx=0.1,ofz=0.3):
  """
  Builds a 2D layered, v(z) fault model.
  Returns the velocity model, reflectivity, fault labels
  and a zero-offset image

  Parameters:
    nz     - number of depth samples [512]
    nx     - number of lateral samples[1000]
    dz     - depth sampling interval [12.5]
    dx     - lateral sampling interval [12.5]
    nlayer - number of deposited layers (there exist many fine layers within a deposit) [21]
    nfx    - number of faults [0.3]
    ofx    - Starting position of faults (percentage of total model) [0.4]
    dx     - Spacing between faults (percentage of total model) [0.1]
    ofz    - Central depth of faults (percentage of total model) [0.3]
    rect   - radius for gaussian smoother [0.5]

  Returns:
    The velocity, reflectivity, fault label and image all of size [nx,nz]
  """
  # Model building object
  mb = mdlbuild.mdlbuild(nx,dx,ny=200,dy=dx,dz=dz,basevel=5000)
  nzi = 1000 # internal size is 1000

  # Propagation velocities
  props = np.linspace(maxvel,minvel,nlayer)

  # Specify the thicknesses
  thicks = np.random.randint(40,61,nlayer)

  dlyr = 0.05
  for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,layer=50,layer_rand=0.00,dev_layer=dlyr)

  # Water deposit
  mb.deposit(1480,thick=80,layer=150,dev_layer=0.0)

  # Trim model before faulting
  mb.trim(0,1100)

  # Put in the faults
  for ifl in progressbar(range(nfx), "nfaults:"):
    x = ofx + ifl*dfx
    mb.fault2d(begx=x,begz=ofz,daz=8000,dz=5000,azim=0.0,theta_die=11,theta_shift=4.0,dist_die=0.3,throwsc=10.0)

  # Get the model
  vel = gaussian_filter(mb.vel[:,:nzi].T,sigma=rect).astype('float32')
  lbl = mb.get_label2d()[:,:nzi].T
  ref = mb.get_refl2d()[:,:nzi].T
  # Parameters for ricker wavelet
  nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
  amp = 1.0; dly = 0.125
  minf = 30.0; maxf = 60.0
  # Create normalized image
  f = rndut.randfloat(minf,maxf)
  wav = ricker(nt,dt,f,amp,dly)
  img = dlut.normalize(np.array([np.convolve(ref[:,ix],wav) for ix in range(nx)])[:,ns:nzi+ns].T)
  nze = dlut.normalize(bandpass(np.random.rand(nzi,nx)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
  img += nze

  # Window the models and return
  f1 = 50
  velwind = vel[f1:f1+nz,:]
  lblwind = lbl[f1:f1+nz,:]
  refwind = ref[f1:f1+nz,:]
  imgwind = img[f1:f1+nz,:]

  return velwind,refwind,imgwind,lblwind

def undulatingfaults2d(nz=512,nx=1000,dz=12.5,dx=25.0,nlayer=21,minvel=1600,maxvel=3000,rect=0.5,
                    nfx=3,ofx=0.4,dfx=0.1,ofz=0.3,noctaves=None,npts=None,amp=None):
  """
  Builds a 2D faulted velocity model with undulating layers
  Returns the velocity model, reflectivity, fault labels
  and a zero-offset image

  Parameters:
    nz       - number of depth samples [512]
    nx       - number of lateral samples[1000]
    dz       - depth sampling interval [12.5]
    dx       - lateral sampling interval [12.5]
    nlayer   - number of deposited layers (there exist many fine layers within a deposit) [21]
    nfx      - number of faults [0.3]
    ofx      - Starting position of faults (percentage of total model) [0.4]
    dx       - Spacing between faults (percentage of total model) [0.1]
    ofz      - Central depth of faults (percentage of total model) [0.3]
    rect     - radius for gaussian smoother [0.5]
    noctaves - octaves perlin parameters for squish [varies between 3 and 6]
    amp      - amplitude of folding [varies between 200 and 500]
    npts     - grid size for perlin noise [3]

  Returns:
    The velocity, reflectivity, fault label and image all of size [nx,nz]
  """
  # Model building object
  mb = mdlbuild.mdlbuild(nx,dx,ny=20,dy=dx,dz=dz,basevel=5000)
  nzi = 1000 # internal size is 1000

  # Propagation velocities
  props = np.linspace(maxvel,minvel,nlayer)

  # Specify the thicknesses
  thicks = np.random.randint(40,61,nlayer)

  dlyr = 0.05
  for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,layer=50,layer_rand=0.00,dev_layer=dlyr)
    if(ilyr == int(nlayer-2)):
      mb.squish(amp=300,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',octaves=3,order=3)

  # Water deposit
  mb.deposit(1480,thick=80,layer=150,dev_layer=0.0)

  # Smooth the interface
  mb.smooth_model(rect1=1,rect2=5,rect3=1)

  # Trim model before faulting
  mb.trim(0,1100)

  # Put in the faults
  for ifl in progressbar(range(nfx), "nfaults:"):
    x = ofx + ifl*dfx
    mb.fault2d(begx=x,begz=ofz,daz=4000,dz=2500,azim=0.0,theta_die=11,theta_shift=6.0,dist_die=2.0,throwsc=10.0)
    mb.fault2d(begx=x,begz=ofz,daz=8000,dz=5000,azim=0.0,theta_die=11,theta_shift=4.0,dist_die=2.0,throwsc=10.0)

  # Get the model
  vel = gaussian_filter(mb.vel[:,:nzi].T,sigma=rect).astype('float32')
  lbl = mb.get_label2d()[:,:nzi].T
  ref = mb.get_refl2d()[:,:nzi].T
  # Parameters for ricker wavelet
  nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
  amp = 1.0; dly = 0.125
  minf = 30.0; maxf = 60.0
  # Create normalized image
  f = rndut.randfloat(minf,maxf)
  wav = ricker(nt,dt,f,amp,dly)
  img = dlut.normalize(np.array([np.convolve(ref[:,ix],wav) for ix in range(nx)])[:,ns:nzi+ns].T)
  nze = dlut.normalize(bandpass(np.random.rand(nzi,nx)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
  img += nze

  # Window the models and return
  f1 = 50
  velwind = vel[f1:f1+nz,:]
  lblwind = lbl[f1:f1+nz,:]
  refwind = ref[f1:f1+nz,:]
  imgwind = img[f1:f1+nz,:]

  return velwind,refwind,imgwind,lblwind

def undulatingfaults_old(nz=512,nx=1000,dz=12.5,dx=25.0,nlayer=21,minvel=1600,maxvel=3000,rect=0.5,
                    nfx=3,ofx=0.4,dfx=0.1,ofz=0.3,noctaves=None,npts=None,amp=None):
  """
  Builds a 2D faulted velocity model with undulating layers
  Returns the velocity model, reflectivity, fault labels
  and a zero-offset image

  Parameters:
    nz       - number of depth samples [512]
    nx       - number of lateral samples[1000]
    dz       - depth sampling interval [12.5]
    dx       - lateral sampling interval [12.5]
    nlayer   - number of deposited layers (there exist many fine layers within a deposit) [21]
    nfx      - number of faults [0.3]
    ofx      - Starting position of faults (percentage of total model) [0.4]
    dx       - Spacing between faults (percentage of total model) [0.1]
    ofz      - Central depth of faults (percentage of total model) [0.3]
    rect     - radius for gaussian smoother [0.5]
    noctaves - octaves perlin parameters for squish [varies between 3 and 6]
    amp      - amplitude of folding [varies between 200 and 500]
    npts     - grid size for perlin noise [3]

  Returns:
    The velocity, reflectivity, fault label and image all of size [nx,nz]
  """
  # Model building object
  mb = mdlbuild.mdlbuild(nx,dx,ny=200,dy=dx,dz=dz,basevel=5000)
  nzi = 1000 # internal size is 1000

  # Propagation velocities
  props = np.linspace(maxvel,minvel,nlayer)

  # Specify the thicknesses
  thicks = np.random.randint(40,61,nlayer)

  dlyr = 0.05
  for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,layer=50,layer_rand=0.00,dev_layer=dlyr)
    if(ilyr == int(nlayer-2)):
      mb.squish(amp=300,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',octaves=3,order=0)

  # Water deposit
  mb.deposit(1480,thick=80,layer=150,dev_layer=0.0)

  # Smooth the interface
  mb.smooth_model(rect1=1,rect2=5,rect3=1)

  # Trim model before faulting
  mb.trim(0,1100)

  # Put in the faults
  for ifl in progressbar(range(nfx), "nfaults:"):
    x = ofx + ifl*dfx
    mb.fault(begx=x,begz=ofz,daz=4000,dz=2500,azim=0.0,theta_die=11,theta_shift=6.0,dist_die=2.0,throwsc=10.0)
    mb.fault(begx=x,begz=ofz,daz=8000,dz=5000,azim=0.0,theta_die=11,theta_shift=4.0,dist_die=2.0,throwsc=10.0)

  # Get the model
  vel = gaussian_filter(mb.vel[100,:,:nzi].T,sigma=rect).astype('float32')
  lbl = mb.get_label()[100,:,:nzi].T
  ref = mb.get_refl()[100,:,:nzi].T
  # Parameters for ricker wavelet
  nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
  amp = 1.0; dly = 0.125
  minf = 30.0; maxf = 60.0
  # Create normalized image
  f = rndut.randfloat(minf,maxf)
  wav = ricker(nt,dt,f,amp,dly)
  img = dlut.normalize(np.array([np.convolve(ref[:,ix],wav) for ix in range(nx)])[:,ns:nzi+ns].T)
  nze = dlut.normalize(bandpass(np.random.rand(nzi,nx)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
  img += nze

  # Window the models and return
  f1 = 50
  velwind = vel[f1:f1+nz,:]
  lblwind = lbl[f1:f1+nz,:]
  refwind = ref[f1:f1+nz,:]
  imgwind = img[f1:f1+nz,:]

  return velwind,refwind,imgwind,lblwind

def undulatingrandfaults2d(nz=512,nx=1000,dz=12.5,dx=25.0,nlayer=21,minvel=1600,maxvel=3000,rect=0.5,
                    nfx=3,ofx=0.4,dfx=0.1,ofz=0.3,noctaves=None,npts=None,amp=None):
  """
  Builds a 2D faulted velocity model with undulating layers
  Returns the velocity model, reflectivity, fault labels
  and a zero-offset image

  Parameters:
    nz       - number of depth samples [512]
    nx       - number of lateral samples[1000]
    dz       - depth sampling interval [12.5]
    dx       - lateral sampling interval [12.5]
    nlayer   - number of deposited layers (there exist many fine layers within a deposit) [21]
    nfx      - number of faults [0.3]
    ofx      - Starting position of faults (percentage of total model) [0.4]
    dx       - Spacing between faults (percentage of total model) [0.1]
    ofz      - Central depth of faults (percentage of total model) [0.3]
    rect     - radius for gaussian smoother [0.5]
    noctaves - octaves perlin parameters for squish [varies between 3 and 6]
    amp      - amplitude of folding [varies between 200 and 500]
    npts     - grid size for perlin noise [3]

  Returns:
    The velocity, reflectivity, fault label and image all of size [nx,nz]
  """
  # Model building object
  # Remember to change dist_die based on ny
  mb = mdlbuild.mdlbuild(nx,dx,ny=20,dy=dx,dz=dz,basevel=5000)
  nzi = 1000 # internal size is 1000

  # Propagation velocities
  props = np.linspace(maxvel,minvel,nlayer)

  # Specify the thicknesses
  thicks = np.random.randint(40,61,nlayer)

  dlyr = 0.05
  for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,layer=50,layer_rand=0.00,dev_layer=dlyr)
    if(ilyr == int(nlayer-2)):
      mb.squish(amp=300,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',octaves=4,order=3)

  # Water deposit
  mb.deposit(1480,thick=80,layer=150,dev_layer=0.0)

  # Smooth the interface
  mb.smooth_model(rect1=1,rect2=5,rect3=1)

  # Trim model before faulting
  mb.trim(0,1100)

  # Thresh should be a function of theta_shift

  # Generate the fault positions
  flttype = np.random.choice([0,1,2,3,4,5])

  if(flttype == 0):
    largefaultblock(mb,0.3,0.7,ofz,nfl=6)
  elif(flttype == 1):
    slidingfaultblock(mb,0.3,0.7,ofz,nfl=6)
  elif(flttype == 2):
    mediumfaultblock(mb,0.3,0.7,0.25,space=0.02,nfl=10)
  elif(flttype == 3):
    mediumfaultblock(mb,0.3,0.7,0.25,space=0.005,nfl=20)
  elif(flttype == 4):
    tinyfaultblock(mb,0.3,0.7,0.25,space=0.02,nfl=10)
  else:
    tinyfaultblock(mb,0.3,0.7,0.25,space=0.005,nfl=20)

  # Get the model
  vel = gaussian_filter(mb.vel[:,:nzi].T,sigma=rect).astype('float32')
  lbl = mb.get_label2d()[:,:nzi].T
  ref = mb.get_refl2d()[:,:nzi].T
  # Parameters for ricker wavelet
  nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
  amp = 1.0; dly = 0.125
  minf = 30.0; maxf = 60.0
  # Create normalized image
  f = rndut.randfloat(minf,maxf)
  wav = ricker(nt,dt,f,amp,dly)
  img = dlut.normalize(np.array([np.convolve(ref[:,ix],wav) for ix in range(nx)])[:,ns:nzi+ns].T)
  nze = dlut.normalize(bandpass(np.random.rand(nzi,nx)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
  img += nze

  # Window the models and return
  f1 = 50
  velwind = vel[f1:f1+nz,:]
  lblwind = lbl[f1:f1+nz,:]
  refwind = ref[f1:f1+nz,:]
  imgwind = img[f1:f1+nz,:]

  return velwind,refwind,imgwind,lblwind

def largefaultblock(mb,minpos,maxpos,ofz,nfl=6):
  lfpars = mb.getfaultpos2d(minpos,maxpos,minblk=0.01,minhor=0.08,mingrb=0.05,nfaults=nfl)
  for ifl in progressbar(range(nfl), "nlfaults:"):
    iflpar = lfpars[ifl]
    zvar = rndut.randfloat(-0.01,0.01)
    z = ofz + zvar
    mb.fault2d(begx=iflpar[0],begz=z,daz=8000,dz=5000,azim=iflpar[1],theta_die=11,theta_shift=3.0,dist_die=2.0,throwsc=10.0,thresh=0.1)

def slidingfaultblock(mb,minpos,maxpos,ofz,nfl=6):
  sfpars = mb.getfaultpos2d(minpos,maxpos,minblk=0.02,minhor=0.2,mingrb=0.2,nfaults=nfl)
  for ifl in progressbar(range(nfl), "nsfaults:"):
    ifspar = sfpars[ifl]
    zvar = rndut.randfloat(-0.01,0.01)
    z = ofz + zvar
    mb.fault2d(begx=ifspar[0],begz=ofz,daz=7500,dz=15000,azim=ifspar[1],theta_die=11,theta_shift=3.0,dist_die=2.0,throwsc=10.0,thresh=0.1)

def mediumfaultblock(mb,minpos,maxpos,ofz,space,nfl=10):
  mfpars = mb.getfaultpos2d(minpos,maxpos,minblk=space,minhor=space,mingrb=space,nfaults=nfl)
  for ifl in progressbar(range(nfl), "nmfaults:"):
    ifmpar = mfpars[ifl]
    zvar = rndut.randfloat(-0.01,0.01)
    z = ofz + zvar
    mb.fault2d(begx=ifmpar[0],begz=z,daz=4000,dz=2500,azim=ifmpar[1],theta_die=11,theta_shift=3.0,dist_die=2.0,throwsc=10.0,thresh=0.2)

def tinyfaultblock(mb,minpos,maxpos,ofz,space,nfl=10):
  tfpars = mb.getfaultpos2d(minpos,maxpos,space,space,space,nfaults=nfl)
  for ifl in progressbar(range(nfl), "ntfaults:"):
    iftpar = tfpars[ifl]
    zvar = rndut.randfloat(-0.01,0.01)
    z = ofz + zvar
    mb.fault2d(begx=iftpar[0],begz=z,daz=2000,dz=1250,azim=iftpar[1],theta_die=11,theta_shift=4.0,dist_die=2.0,throwsc=10.0,thresh=0.15)

