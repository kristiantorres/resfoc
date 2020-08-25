"""
Creates heavily faulted and folded pseudo random velocity models.
Based on the software by Bob Clapp

@author: Joseph Jennings
@version: 2020.04.20
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import velocity.mdlbuild as mdlbuild
from scaas.wavelet import ricker
from scaas.trismooth import smooth
from genutils.ptyprint import progressbar, create_inttag
import genutils.rand as rndut
import deeplearn.utils as dlut
from scipy.ndimage import gaussian_filter
from genutils.signal import bandpass

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
sep = seppy.sep()

## Get commandline arguments
# Inputs and outputs
beg     = args.beg
end     = args.end
outdir  = args.outdir
outmod  = outdir + "/" + "velfltmod" + create_inttag(beg,end) + ".H"
outlbl  = outdir + "/" + "velfltlbl" + create_inttag(beg,end) + ".H"
outref  = outdir + "/" + "velfltref" + create_inttag(beg,end) + ".H"
outimg  = outdir + "/" + "velfltimg" + create_inttag(beg,end) + ".H"
prefix  = args.prefix
nmodels = args.nmodels

# Get the parameters of the model
nx = args.nx; dx = args.dx
ny = args.ny; dy = args.dy
nz = args.nz; dz = args.dz

# Parameters for ricker wavelet
nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
amp = 1.0; dly = 0.125
minf = 30.0; maxf = 60.0

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
  nlayer = 21
  #minvel = 1600; maxvel = 5000
  #props = mb.vofz(nlayer,minvel,maxvel)

  #nlayer = 150
  minvel = 1600; maxvel = 3000
  props = np.linspace(maxvel,minvel,nlayer)

  # Specify the thicknesses
  thicks = np.random.randint(40,61,nlayer)

  # Determine when to fold the deposits
  sqlyrs = sorted(mb.findsqlyrs(3,nlayer,5))
  csq = 0

  dlyr = 0.05
  for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],band2=0.01,band3=0.05,dev_pos=0.0,layer=50,layer_rand=0.00,dev_layer=dlyr)

  # Water deposit
  mb.deposit(1480,thick=80,layer=150,dev_layer=0.0)
  # Trim model before faulting
  mb.trim(0,1100)

  # Fault it up!
  azims = [0.0,180.0]

  ox = 0.4; dx = 0.1
  for ifl in progressbar(range(3), "nfaults:"):
    x = ox + ifl*dx
    mb.fault2d(begx=x,begz=0.3,daz=8000,dz=5000,azim=0.0,
      theta_die=11,theta_shift=4.0,dist_die=0.3,throwsc=10.0)
    #mb.smallfault(azim=0.0,begz=0.35,begx=x,tscale=5.0,twod=True)

  # Get model
  if(nxo == nx and nzo == nz):
    vels[:,:,imodel] = gaussian_filter(mb.vel[:,:nz].T,sigma=args.rect).astype('float32')
    lbls[:,:,imodel] = mb.get_label2d()[:,:nz].T
    refs[:,:,imodel] = mb.get_refl2d()[:,:nz].T
    # Create normalized image
    f = rndut.randfloat(minf,maxf)
    wav = ricker(nt,dt,f,amp,dly)
    img = dlut.normalize(np.array([np.convolve(refs[:,ix,imodel],wav) for ix in range(nx)])[:,ns:1000+ns].T)
    # Create noise
    nze = dlut.normalize(bandpass(np.random.rand(nz,nx)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
    imgs[:,:,imodel] = img + nze
  else:
    vels[:,:,imodel] = dlut.resample(mb.vel[slcy,:,:nz],[nxo,nzo],kind='linear')
    lbls[:,:,imodel] = dlut.resample(mb.get_label()[slcy,:,:nz],[nxo,nzo],kind='linear')
    refs[:,:,imodel] = dlut.resample(mb.get_refl()[slcy,:,:nz].T,[nxo,nzo],kind='linear')
    # Create normalized image
    f = rndut.randfloat(minf,maxf)
    wav = ricker(nt,dt,f,amp,dly)
    img = dlut.normalize(np.array([np.convolve(refs[:,ix,imodel],wav) for ix in range(nx)])[:,ns:1000+ns].T)
    # Create noise
    nze = dlut.normalize(bandpass(np.random.rand(nzo,nxo)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
    imgs[:,:,imodel] = img + nze

# Window the arrays
f2=0; n2=1000; f1=50; n1=512
#f2=250; n2=512; f1=50; n1=512
velwind = vels[f1:f1+n1,f2:f2+n2]
lblwind = lbls[f1:f1+n1,f2:f2+n2]
refwind = refs[f1:f1+n1,f2:f2+n2]
imgwind = imgs[f1:f1+n1,f2:f2+n2]

# Write the velocity model and the label
ds = [10.0,10.0]
sep.write_file("velbig.H",velwind,ds=ds,dpath=args.datapath)
sep.write_file("lblbig.H",lblwind,ds=ds,dpath=args.datapath)
sep.write_file("refbig.H",refwind,ds=ds,dpath=args.datapath)
sep.write_file("imgbig.H",imgwind,ds=ds,dpath=args.datapath)

# Flag for cluster manager to determine success
print("Success!")
