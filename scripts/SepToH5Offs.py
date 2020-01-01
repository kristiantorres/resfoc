"""
Converts prestack images in SEPlib
format to H5 format. Takes only the rho=1 residual migration
image and all of the subsurface offset images

User must specify the directory in which the data
are located and the output .h5 file.

@author: Joseph Jennings
@version: 2019.12.31
"""

import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import subprocess, glob
import h5py

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "sepdir": None,
    "verb": "y",
    "ro1idx": 5,
    "nprint": 100,
    }
if args.conf_file:
  config = ConfigParser.SafeConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("Defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

# Set defaults
parser.set_defaults(**defaults)

# Input files
parser.add_argument("-sepdir",help="Directory containing SEPlib files",required=True,type=str)
parser.add_argument("-out",help="output h5 file",required=True,type=str)
# Optional arguments
parser.add_argument("-ro1idx",help="Index of rho=1 [5]",type=int)
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-nprint",help="Print after so many examples [100]",type=int)
args = parser.parse_args(remaining_argv)

# Set up IO
sep = seppy.sep(sys.argv)

# Get command line arguments
sepdir = args.sepdir
out = args.out
ro1idx = args.ro1idx
verb = sep.yn2zoo(args.verb)
nprint = args.nprint

# First get total number of examples
imgfiles = sorted(glob.glob(sepdir + '/img*.H'))
lblfiles = sorted(glob.glob(sepdir + '/lbl*.H'))

assert(len(imgfiles) == len(lblfiles)), "Must have as many features as labels. Exiting"

# Get total number of examples
ntot = 0
for ifile in imgfiles: ntot += (sep.read_header(None,ifname=ifile)).n[4]

if(verb): print("Total number of examples: %d"%(ntot))

# Allocate output array
iaxes = sep.read_header(None,ifname=imgfiles[0])
nz = iaxes.n[0]; nx = iaxes.n[1]; nh = iaxes.n[2];

imgs = np.zeros([nz,nx,nh,ntot],dtype='float32')
rhos = np.zeros([nz,nx,ntot],dtype='float32')

beg = 0; end = 0; k = 0
for ifile in zip(imgfiles,lblfiles):
  if(verb):
    if(k%nprint == 0):
      print("%d ... "%(k),end=''); sys.stdout.flush()
  # Read in image
  iaxes,img = sep.read_file(None,ifname=ifile[0])
  img = img.reshape(iaxes.n,order='F')
  # Read in rho
  raxes,rho = sep.read_file(None,ifname=ifile[1])
  rho = rho.reshape(raxes.n,order='F')
  # Save to output arrays
  end += iaxes.n[4]
  imgs[:,:,:,beg:end] = img[:,:,:,ro1idx,:]
  rhos[:,:,beg:end] = rho[:,:,:]
  # Update counters
  beg = end; k += 1

# Transpose
imgst = np.transpose(imgs,(3,0,1,2))
rhost = np.transpose(rhos,(2,0,1))
with h5py.File(out,'w') as hf:
  hf.create_dataset("data", (ntot,nz,nx,nh), data=imgst, dtype=np.float32)
  hf.create_dataset("labels", (ntot,nz,nx), data=rhost, dtype=np.float32)

