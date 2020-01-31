"""
Creates point scatterer training data for deep learning residual migration.
The idea is to replicate (with deep learning) the experiments performed in the paper:
Velocity estimation by image-focusing analysis, Biondi 2010.

Output training data is in H5 format of training data pairs. The features (X)
consist of prestack residual migration images (Z,X,OFFSET,RHO) and the labels (Y) consist
of the rho value to be estimated (Z,X). Note that the number of subsurface offsets (nh)
gives the nh provided as the zero subsurface offset and the actual output
of subsurface offsets as 2*nh+1 (this forces symmetry about zero subsurface offset)

For imaging, a split spread acquisition is always assumed where receivers are placed
at each gridpoint and sources are placed at every 10 gridpoints. Default sampling
intervals are dx=20m and dz=20m

The wavelet used for modeling is a ricker with central frequency of 15Hz (max around 30Hz)

The migration velocity is always constant (v=2500 m/s) and the modeling
velocity varies around 2500 m/s determined by the rho axis chosen (nro,oro,dro)

For the residual migration, the nro provided is just one side of the
residual migration. So the actual output of residual migration parameters is
2*nro-1 (again to enforce symmetry) and therefore the actual oro is computed
as: oro - (nro-1)*dro. This forces that the output be centered at the oro
provided by the user

@author: Joseph Jennings
@version: 2019.12.22
"""

import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from resfoc.training_data import createdata_ptscat
import h5py
import time

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "nex": 2000,
    "split": 1.0,
    "nsx": 26,
    "osx": 3,
    "nz": 256,
    "nx": 256,
    "nh": 10,
    "nro": 6,
    "oro": 1.0,
    "dro": 0.01,
    "verb": 'y',
    "keepoff": 'n',
    "nprint": 100
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
# Output H5 file
ioArgs = parser.add_argument_group('Output files')
ioArgs.add_argument("-outtr",help="Output training data (X,Y in h5 format)")
ioArgs.add_argument("-outva",help="Output validation data (not output if split is 1.0)")
# Imaging parameters
imgArgs = parser.add_argument_group('Imaging parameters')
imgArgs.add_argument("-nsx",help="Number of sources [41]",type=int)
imgArgs.add_argument("-osx",help="First source point [0 samples]",type=int)
imgArgs.add_argument("-nz",help="Number of depth samples of image [256]",type=int)
imgArgs.add_argument("-nx",help="Number of lateral samples of image [500]",type=int)
imgArgs.add_argument("-nh",help="Number of subsurface offsets of image [10]",type=int)
# Residual migration parameters
rmigArgs = parser.add_argument_group('Residual migration parameters')
rmigArgs.add_argument("-nro",help="Number of residual migrations [6]",type=int)
rmigArgs.add_argument("-oro",help="Center residual migration [1.0]",type=float)
rmigArgs.add_argument("-dro",help="Rho spacing [0.01]",type=float)
# Machine learning parameters
mlArgs = parser.add_argument_group('Machine learning parameters')
mlArgs.add_argument("-nex",help="Total number of examples [2000]",type=int)
mlArgs.add_argument("-split",help="Training/validation split [1.0]",type=float)
mlArgs.add_argument("-keepoff",help="Keep the subsurface offsets in the training data (y or [n])",type=str)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-verb",help="Verbosity flag [y]",type=str)
othArgs.add_argument("-nprint",help="How often to print a new example [100]",type=int)
args = parser.parse_args(remaining_argv)

# Setup IO
sep = seppy.sep(sys.argv)

# Verbosity argument
verb = sep.yn2zoo(args.verb)
nprint = args.nprint

# Imaging parameters
nsx = args.nsx; osx = args.osx
nz = args.nz; nx = args.nx; nh = args.nh

# Residual migration parameters
nro = args.nro; oro = args.oro; dro = args.dro

# Machine learning parameters
nex = args.nex; split = args.split
keepoff = sep.yn2zoo(args.keepoff)

# Output arrays
allimgs = None
if(keepoff):
  allimgs = np.zeros([nex,nz,nx,2*nh+1,2*nro-1],dtype='float32')
else:
  allimgs = np.zeros([nex,nz,nx,2*nro-1],dtype='float32')
allrhos = np.zeros([nex,nz,nx],dtype='float32')

# Compute split
ntrn = int(nex*args.split)
nval = nex - ntrn

if(verb):
  print("Total number of examples: %d"%(nex))
  print("Number of training: %d"%(ntrn))
  print("Number of development: %d"%(nval))

# Create all examples
for iex in range(nex):
  if(verb):
    if(iex%nprint == 0):
      print("%d ... "%(iex),end='')
  if(keepoff):
    allimgs[iex,:,:,:,:], allrhos[iex,:,:] = createdata_ptscat(nsx,osx,nz,nx,nh,nro,oro,dro,keepoff=keepoff,debug=False)
  else:
    allimgs[iex,:,:,:],   allrhos[iex,:,:] = createdata_ptscat(nsx,osx,nz,nx,nh,nro,oro,dro,keepoff=keepoff,debug=False)

print(" ")

# Write out data in h5 format
if(nval == 0):
  with h5py.File(args.outtr,'w') as hf:
    # Features
    if(keepoff):
      hf.create_dataset("data", (ntrn,nz,nx,2*nh+1,2*nro-1), data=allimgs, dtype=np.float32)
    else:
      hf.create_dataset("data", (ntrn,nz,nx,2*nro-1), data=allimgs, dtype=np.float32)
    # Labels
    hf.create_dataset("label", (ntrn,nz,nx), data=allrhos, dtype=np.float32)
else:
  with h5py.File(args.outtr,'w') as hf:
    # Features
    if(keepoff):
      hf.create_dataset("data", (ntrn,nz,nx,2*nh+1,2*nro-1), data=allimgs[:ntrn,:,:,:,:], dtype=np.float32)
    else:
      hf.create_dataset("data", (ntrn,nz,nx,2*nro-1), data=allimgs[:ntrn,:,:,:], dtype=np.float32)
    # Labels
    hf.create_dataset("label", (ntrn,nz,nx), data=allrhos[:ntrn,:,:], dtype=np.float32)
  with h5py.File(args.outva,'w') as hf:
    # Features
    if(keepoff):
      hf.create_dataset("data", (nval,nz,nx,2*nh+1,2*nro-1), data=allimgs[nval:,:,:,:,:], dtype=np.float32)
    else:
      hf.create_dataset("data", (nval,nz,nx,2*nro-1), data=allimgs[nval:,:,:], dtype=np.float32)
    # Labels
    hf.create_dataset("label", (nval,nz,nx), data=allrhos[nval:,:,:], dtype=np.float32)

