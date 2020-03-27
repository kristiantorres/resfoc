"""
Creates an H5 file with 2D velocity, migration velocity and reflectivity models

Each H5 dataset is a 3D array with nz,nx and then three channels
for true velocity, migration velocity and reflectivity

@author: Joseph Jennings
@version: 2020.01.07
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from random import choice
from scipy.ndimage import gaussian_filter
from deeplearn.utils import resample
import glob
import h5py

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    "nz": 256,
    "nx": 400,
    "rad": 10,
    "nprint": 100,
    "prefix": "",
    }
if args.conf_file:
  config = configparser.ConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

# Set defaults
parser.set_defaults(**defaults)

# Input files
ioArgs = parser.add_argument_group('Inputs and outputs')
ioArgs.add_argument("-path",help="Path to velocity models",type=str)
ioArgs.add_argument("-out",help="Output H5 file",type=str)
ioArgs.add_argument("-prefix",help="Prefix for all velocity models [None]",type=str)
# Optional arguments
velArgs = parser.add_argument_group("Velocity model processing arguments")
velArgs.add_argument("-nx",help="Output lateral number of samples [400]",type=int)
velArgs.add_argument("-nz",help="Output depth number of samples [256]",type=int)
velArgs.add_argument("-rad",help="Smoothing radius for creating migration velocity [10]",type=int)
# Other arguments
parser.add_argument("-nprint",help="Print after so many files [100]",type=int)
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Input output arguments
path   = args.path
out    = args.out
prefix = args.prefix
# Processing arguments
nzo = args.nz; nxo = args.nx
rad = args.rad
# Other arguments
verb = sep.yn2zoo(args.verb)

# Get the velocity models
velfiles = sorted(glob.glob(path + '/' + prefix + '*.H'))
newidx = 2170 + 974
velfiles = velfiles[2170:]
nvels = len(velfiles)

# Open the output H5 file
hfout = h5py.File(out,'w')

# Temporary arrays array to hold parameters
vout = np.zeros([3,nzo,nxo],dtype='float32')

# Random options
slcx = [True,False]

# Process and save each velocity model
k = newidx
for ifile in velfiles:
  if(verb): print("%d/%d models complete"%(k,nvels),end='\r')
  # Read in the velocity model
  vaxes,vel = sep.read_file(None,ifname=ifile)
  vel = vel.reshape(vaxes.n,order='F')
  # Grab a random slice along x or y
  if(choice(slcx)):
    idxx = choice(list(range(vaxes.n[1])))
    vbig = vel[:,idxx,:]
  else:
    idxy = choice(list(range(vaxes.n[2])))
    vbig = vel[:,idxy,:]
  # Interpolate the model
  vout[0,:,:] = (resample(vbig.T,[nxo,nzo])).T
  # Smooth the interpolated model
  vout[1,:,:] = gaussian_filter(vout[0,:,:],sigma=rad)
  # Compute the reflectivity
  vout[2,:,:] = vout[0,:,:] - vout[1,:,:]
  # Save as a H5 dataset
  hfout.create_dataset(sep.create_inttag(k,nvels),(3,nzo,nxo), data=vout, dtype=np.float32)
  k += 1

# Close the file
hfout.close()

