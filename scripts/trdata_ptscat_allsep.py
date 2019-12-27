"""
Creates point scatterer training data for deep learning residual migration.
The idea is to replicate (with deep learning) the experiments performed in the paper:
Velocity estimation by image-focusing analysis, Biondi 2010.

Outputs the labels and the features as separate SEPlib files. If a large number of
examples are desired, the program will write a file after every 50 examples have
been obtained. The output of each label file is the rho field as a function of
x and z. The features output are residually migrated prestack images.

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
@version: 2019.12.26
"""

import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from training_data import createdata_ptscat
import h5py
import time

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "nex": 2000,
    "nwrite": 20,
    "nsx": 26,
    "osx": 3,
    "nz": 256,
    "nx": 256,
    "nh": 10,
    "nro": 6,
    "oro": 1.0,
    "dro": 0.01,
    "verb": 'y',
    "nprint": 100,
    "prefix": "",
    "beg": 0,
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
# IO
ioArgs = parser.add_argument_group('Output files')
ioArgs.add_argument("-outdir",help="Output directory of where to write the SEPlib files",type=str)
ioArgs.add_argument("-datapath",help="Output datapath of where to write the SEPlib binaries",type=str)
ioArgs.add_argument("-nwrite",help="Number of examples to compute before writing [20]",type=int)
ioArgs.add_argument("-prefix",help="Prefix that will be used for label and feature files [None]",type=str)
ioArgs.add_argument("-beg",help="Numeric suffix used for keeping track of examples [0]",type=int)
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
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-verb",help="Verbosity flag [y]",type=str)
othArgs.add_argument("-nprint",help="How often to print a new example [100]",type=int)
args = parser.parse_args(remaining_argv)

# Setup IO
sep = seppy.sep(sys.argv)

# IO parameters
outdir = args.outdir; nwrite = args.nwrite
prefix = args.prefix; dpath = args.datapath
beg = args.beg

# Verbosity argument
verb = sep.yn2zoo(args.verb)
nprint = args.nprint

# Imaging parameters
nsx = args.nsx; osx = args.osx
nz = args.nz; nx = args.nx; nh = args.nh

# Residual migration parameters
nro = args.nro; oro = args.oro; dro = args.dro

# Machine learning parameters
nex = args.nex

# Output arrays and axes
dz = 20.0; dx = 20.0
iaxes = seppy.axes([nz,nx,2*nh+1,2*nro-1,nwrite],[0.0,0.0,-nh*dx,oro-(nro-1)*dro,0.0],[dz,dx,dx,dro,1.0])
raxes = seppy.axes([nz,nx,nwrite],[0.0,0.0,0.0],[dz,dx,1.0])
imgs = np.zeros([nz,nx,2*nh+1,2*nro-1,nwrite],dtype='float32')
rhos = np.zeros([nz,nx,nwrite],dtype='float32')

# Create all examples
k = 0
for iex in range(nex):
  if(verb):
    if(iex%nprint == 0):
      print("%d ... "%(iex),end='')
    imgs[:,:,:,:,k], rhos[:,:,k] = createdata_ptscat(nsx,osx,nz,nx,nh,nro,oro,dro,keepoff=True,debug=False)
  if(k == nwrite-1):
    # Create tag
    suffix = sep.create_inttag(beg+iex,nex)+".H"
    # Write features
    sep.write_file(None,iaxes,imgs,ofname=outdir+prefix+"img"+suffix,dpath=dpath)
    # Write labels
    sep.write_file(None,raxes,rhos,ofname=outdir+prefix+"lbl"+suffix,dpath=dpath)
    # Reset arrays
    imgs[:] = 0.0; rhos[:] = 0.0
    if(verb): print("Writing %d examples"%(nwrite))
    # Reset counter
    k = -1
  k += 1

# Write remaining examples
nres = nex%nwrite
if(nres != 0):
  if(verb): print("Writing %d residual examples"%(nres))
  # Residual axes
  resiaxes = seppy.axes([nz,nx,2*nh+1,2*nro-1,nres],[0.0,0.0,-nh*dx,oro-(nro-1)*dro,0.0],[dz,dx,dx,dro,1.0])
  resraxes = seppy.axes([nz,nx,nres],[0.0,0.0,0.0],[dz,dx,1.0])
  # Residual training data
  resimgs = np.zeros([nz,nx,2*nh+1,2*nro-1,nres],dtype='float32')
  resrhos = np.zeros([nz,nx,nres],dtype='float32')
  resimgs[:] = imgs[:,:,:,:,:nres]; resrhos = rhos[:,:,:nres]
  # Write features
  suffix = sep.create_inttag(beg+nex-1,nex) + ".H"
  sep.write_file(None,resiaxes,resimgs,ofname=outdir+prefix+"img"+suffix)
  # Write labels
  sep.write_file(None,resraxes,resrhos,ofname=outdir+prefix+"lbl"+suffix)

print(" ")

