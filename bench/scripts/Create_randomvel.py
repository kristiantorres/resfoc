"""
Creates random, geologically feasible  2D velocity models
using the software developed by Bob Clapp.

@author: Joseph Jennings
@version: 2020.01.07
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import scipy.ndimage as flt
import velocity.pySepVector
import velocity.syntheticModel as syntheticModel
from velocity.structure import vel_structure
from deeplearn.utils import resample
import csv

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "nx": 600,
    "ox": 0.0,
    "dx": 25.0,
    "ny": 600,
    "oy": 0.0,
    "dy": 25.0,
    "nz": 600,
    "oz": 0.0,
    "dz": 10.0,
    "rect": 0.8,
    "nmodels": 1,
    "verb": "y",
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
velArgs = parser.add_argument_group("Velocity model parameters")
velArgs.add_argument("-nx",help="Number of x samples [600]",type=int)
velArgs.add_argument("-ox",help="x origin [0.0]",type=float)
velArgs.add_argument("-dx",help="x sample spacing [25.0]",type=float)
velArgs.add_argument("-ny",help="Number of y samples [600]",type=int)
velArgs.add_argument("-oy",help="y origin [0.0]",type=float)
velArgs.add_argument("-dy",help="y sample spacing [25.0]",type=float)
velArgs.add_argument("-nz",help="Number of z samples [600]",type=int)
velArgs.add_argument("-oz",help="z origin [0.0]",type=float)
velArgs.add_argument("-dz",help="z sample spacing [10.0]",type=float)
prcArgs = parser.add_argument_group("Velocity model processing")
prcArgs.add_argument("-nzo",help="Ouput number of depth samples for interpolation [256]",type=int)
prcArgs.add_argument("-nxo",help="Ouput number of lateral samples for interpolation [400]",type=int)
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
out     = outdir + "/" + "velmod" + sep.create_inttag(beg,end) + ".H"
par     = outdir + "/" + "velpar" + sep.create_inttag(beg,end) + ".par"
prefix  = args.prefix
nmodels = args.nmodels

# Velocity parameters
nx = args.nx; ox = args.ox; dx = args.dx
ny = args.ny; oy = args.oy; dy = args.dy
nz = args.nz; oz = args.oz; dz = args.dz

# Processing
nzo = args.nzo; nxo = args.nxo
rect = args.rect

# Verbosity flag
verb = sep.yn2zoo(args.verb)

### Set the random parameters for building the model
## Define propagation velocities
props = np.zeros(4)
# Bottom layer
prop1min = 4500
prop1max = 5500
props[0] = np.random.rand()*(prop1max - prop1min) + prop1min

# Second layer
prop2min = 3500
prop2max = 4500
props[1] = np.random.rand()*(prop2max - prop2min) + prop2min

# Third layer
prop3min = 2500
prop3max = 3500
props[2] = np.random.rand()*(prop3max - prop3min) + prop3min

# Fourth layer
prop4min = 1500
prop4max = 2500
props[3] = np.random.rand()*(prop4max - prop4min) + prop4min

# Create the outputs
pars = []; vels = np.zeros([nmodels,nzo,nxo],dtype='float32')

## Loop over models
for imodel in range(nmodels):
  stctr = vel_structure(nx); pars.append(stctr)
  if(verb): print("Model %d"%(imodel))
  # Build the actual velocity model
  mod = syntheticModel.geoModel(nx=nx,ox=ox,dx=dx,ny=ny,oy=oy,dy=dy,dz=dz,basement=5000)
  for ilyr in range(4):
    if(verb): print("Layer %d"%(ilyr))
    tag = str(ilyr+1)
    # Deposit
    if(verb): print("Depositing")
    mod.deposit(prop=props[ilyr],thick=stctr['thick'+tag],var=.3,dev_pos=.1,layer=25,dev_layer=.3,layer_rand=.3,band2=.01, band3=0.05)
    # Fold
    if(verb): print("Folding")
    mod.squish(max=stctr['amp'+tag],random_inline=2.,random_crossline=3.,aziumth=stctr['az'+tag],wavelength=stctr['wav'+tag])
    # Faults
    if(verb): print("Faulting")
    nfaults = stctr['nfaults'+tag]
    for i in range(nfaults):
      if(verb): print("%d ... "%(i),end='',flush=True)
      mod.fault(begx=i/float(nfaults),begy=i/float(nfaults),begz=stctr['begzf'+tag],daz=8000,dz=7000,azimuth=stctr['faz'+tag],theta_die=12,
          theta_shift=stctr['theta_shift'+tag],dist_die=.3,perp_die=.5)
    print(" ")

  # Water layer deposit
  if(verb): print("Water layer")
  mod.deposit(prop=1500,thick=10,dev_layer=0, layer_rand=0,layer=100,dev_pos=0.00)

  # Extract values from hypercube
  vel = np.asarray(mod.getProp('velocity'))
  vel = np.transpose(vel, (2,0,1))

  # Take a slice
  if(stctr['choice']):
    velwind = vel[0:nz,stctr['idx'],:]
  else:
    velwind = vel[0:nz,:,stctr['idx']]

  # Interpolate to output
  velre = (resample(velwind.T,[nxo,nzo])).T

  # Smooth and save to output
  vels[imodel,:,:] = flt.gaussian_filter(velre,sigma=rect)

  if(verb): print("Actual nz=%d"%(vel.shape[0]))

# Write the velocity
velst = np.transpose(vels,(1,2,0))
vaxes = seppy.axes([nzo,nxo,nmodels],[0.0,ox,0.0],[dz,dx,1.0])
sep.write_file(None,vaxes,velst,ofname=out,dpath=args.datapath)

# Write the structure parameters
keys = pars[0].keys()
with open(par,'w') as csvfile:
  dict_writer = csv.DictWriter(csvfile, keys)
  dict_writer.writeheader()
  dict_writer.writerows(pars)

