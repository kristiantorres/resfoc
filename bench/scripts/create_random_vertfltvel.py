"""
Creates heavily faulted and folded pseudo random velocity models.
Based on the software by Bob Clapp

@author: Joseph Jennings
@version: 2020.02.20
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import velocity.mdlbuild as mdlbuild
from scaas.wavelet import ricker
from utils.ptyprint import progressbar
import utils.rand as rndut
import deeplearn.utils as dlut
from scipy.ndimage import gaussian_filter
from utils.signal import bandpass
import matplotlib.pyplot as plt
from matplotlib import colors

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "nx": 1000,
    "ox": 0.0,
    "dx": 25.0,
    "ny": 200,
    "oy": 0.0,
    "dy": 25.0,
    "nz": 1000,
    "oz": 0.0,
    "dz": 12.5,
    "rect": 0.5,
    "nmodels": 1,
    "verb": "y",
    "nxo": 1000,
    "nzo": 1000,
    "slcy": 100,
    }
if args.conf_file:
  config = configparser.ConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.set_defaults(**defaults)

# Input/output
ioArgs = parser.add_argument_group("Input output arguments")
ioArgs.add_argument("-outdir",help="Output directory for header and par files",type=str)
ioArgs.add_argument("-datapath",help="Output directory for SEP data files",type=str)
ioArgs.add_argument("-beg",help="Beginning velocity model number",type=int)
ioArgs.add_argument("-end",help="Ending velocity model number",type=int)
ioArgs.add_argument("-prefix",help="Prefix for writing velocity models",type=str)
ioArgs.add_argument("-nmodels",help="Number of output models",type=int)
# Velocity model arguments
velArgs = parser.add_argument_group("Velocity model parameters")
velArgs.add_argument("-nx",help="Number of x samples [1000]",type=int)
velArgs.add_argument("-ox",help="x origin [0.0]",type=float)
velArgs.add_argument("-dx",help="x sample spacing [25.0]",type=float)
velArgs.add_argument("-ny",help="Number of y samples [1000]",type=int)
velArgs.add_argument("-oy",help="y origin [0.0]",type=float)
velArgs.add_argument("-dy",help="y sample spacing [25.0]",type=float)
velArgs.add_argument("-nz",help="Number of z samples [1000]",type=int)
velArgs.add_argument("-oz",help="z origin [0.0]",type=float)
velArgs.add_argument("-dz",help="z sample spacing [12.5]",type=float)
# Processing arguments
prcArgs = parser.add_argument_group("Velocity model processing")
prcArgs.add_argument("-nzo",help="Ouput number of depth samples for interpolation [1000]",type=int)
prcArgs.add_argument("-nxo",help="Ouput number of lateral samples for interpolation [1000]",type=int)
prcArgs.add_argument("-slcy",help="Index at which to slice the velocity model in y [100]",type=int)
prcArgs.add_argument("-rect",help="Window radius for smoother [0.5]",type=float)
parser.add_argument("-verb",help="Verbosity flag ([y] or n)")
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

## Get commandline arguments
# Inputs and outputs
beg     = args.beg
end     = args.end
outdir  = args.outdir
outmod  = outdir + "/" + "velvertfltmod" + sep.create_inttag(beg,end) + ".H"
outlbl  = outdir + "/" + "velvertfltlbl" + sep.create_inttag(beg,end) + ".H"
outref  = outdir + "/" + "velvertfltref" + sep.create_inttag(beg,end) + ".H"
outimg  = outdir + "/" + "velvertfltimg" + sep.create_inttag(beg,end) + ".H"
prefix  = args.prefix
nmodels = args.nmodels

# Get the parameters of the model
nx = args.nx; dx = args.dx
ny = args.ny; dy = args.dy
nz = args.nz; dz = args.dz

# Parameters for ricker wavelet
nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
amp = 1.0; dly = 0.125
minf = 60.0; maxf = 80.0

# Output nz and nx
nzo = args.nzo; nxo = args.nxo
slcy = args.slcy

# Output velocity and labels
vels = np.zeros([nzo,nxo,nmodels],dtype='float32')
lbls = np.zeros([nzo,nxo,nmodels],dtype='float32')
refs = np.zeros([nzo,nxo,nmodels],dtype='float32')
imgs = np.zeros([nzo,nxo,nmodels],dtype='float32')

# Loop over models
for imodel in range(nmodels):
  print("Model %d/%d"%(imodel+1,nmodels))

  # Model building object
  mb = mdlbuild.mdlbuild(nx,dx,ny,dy,dz,basevel=5000)

  # First build the v(z) model
  #TODO: might need to make these parameters
  nlayer = 20
  minvel = 1600; maxvel = 5000
  props = mb.vofz(nlayer,minvel,maxvel)

  # Specify the thicknesses
  thicks = np.random.randint(40,61,nlayer)

  # Determine when to fold the deposits
  sqlyrs = sorted(mb.findsqlyrs(3,nlayer,5))
  csq = 0

  dlyr = 0.1
  for ilyr in progressbar(range(nlayer), "ndeposit", 40):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],band2=0.01,band3=0.05,dev_pos=0.0,layer=150,layer_rand=0.00,dev_layer=dlyr)
    # Random folding
    if(ilyr in sqlyrs):
      if(sqlyrs[csq] < 15):
        # Random amplitude variation in the folding
        amp = np.random.rand()*(1800-500) + 500
        mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
      elif(sqlyrs[csq] >= 15 and sqlyrs[csq] < 18):
        amp = np.random.rand()*(1800-500) + 500
        mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
      else:
        amp = np.random.rand()*(500-300) + 300
        mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
      csq += 1

  # Water deposit
  mb.deposit(1480,thick=50,layer=150,dev_layer=0.0)
  # Trim model before faulting
  mb.trim(0,1100)

  azims = [0.0,180.0]

  # Only vertical faults
  nb = 10
  # Create x and z pairs of where faults were appear
  poss = mb.find_faultpos(nb,0.25)
  for ipos in poss:
    nfl = np.random.randint(5,8)
    azim = np.random.choice(azims)
    mb.verticalfault_block(nfault=nfl,azim=azim,begz=ipos[0],begx=ipos[1],begy=0.5,tscale=3.0,rand=True)

  # Get model
  if(nxo == nx and nzo == nz):
    vels[:,:,imodel] = gaussian_filter(mb.vel[slcy,:,:nz].T,sigma=args.rect).astype('float32')
    lbls[:,:,imodel] = mb.get_label()[slcy,:,:nz].T
    refs[:,:,imodel] = mb.get_refl()[slcy,:,:nz].T
    # Create normalized image
    f = rndut.randfloat(minf,maxf)
    wav = ricker(nt,dt,f,amp,dly)
    img = dlut.normalize(np.array([np.convolve(refs[:,ix,imodel],wav) for ix in range(nx)])[:,ns:1000+ns].T)
    # Create noise
    nze = dlut.normalize(bandpass(np.random.rand(nz,nx)*2-1, 2.0, 0.01, 2, pxd=43))/10.0
    imgs[:,:,imodel] = img + nze
  else:
    velsm = gaussian_filter(mb.vel[slcy,:,:nz],sigma=0.5)
    vels[:,:,imodel] = dlut.resample(velsm,[nxo,nzo],kind='linear').astype('float32').T
    lbls[:,:,imodel] = dlut.thresh(dlut.resample(mb.get_label()[slcy,:,:nz],[nxo,nzo],kind='linear'),0.0).T
    ref = mb.get_refl()[slcy,:,:nz]
    refs[:,:,imodel] = dlut.resample(ref,[nxo,nzo],kind='linear').T
    # Create normalized image
    f = rndut.randfloat(minf,maxf)
    wav = ricker(nt,dt,f,amp,dly)
    img = np.array([np.convolve(ref[ix,:],wav) for ix in range(nx)])[:,ns:1000+ns]
    imgo = dlut.normalize(dlut.resample(img,[nxo,nzo],kind='linear')).T
    # Create noise
    nze = dlut.normalize(bandpass(np.random.rand(nzo,nxo)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
    imgs[:,:,imodel] = imgo + nze

# Write the velocity model and the label
maxes = seppy.axes([nz,nx,nmodels],[0.0,0.0,0.0],[dz,dx,1.0])
sep.write_file(None,maxes,vels,ofname=outmod,dpath=args.datapath)
sep.write_file(None,maxes,lbls,ofname=outlbl,dpath=args.datapath)
sep.write_file(None,maxes,refs,ofname=outref,dpath=args.datapath)
sep.write_file(None,maxes,imgs,ofname=outimg,dpath=args.datapath)

# Flag for cluster manager to determine success
print("Success!")

