"""
Performs a quality-check on the residual migration
training data. Using Vplot, plots random examples.

Because this program uses Vplot, you must have
SEPlib correctly installed and configured
to run

@author: Joseph Jennings
@version: 2020.01.05
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
    "nplots": 2,
    "zidx": 10,
    "dsize": 20,
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
parser.add_argument("-h5in",help="Input h5 file",type=str)
# Optional arguments
parser.add_argument("-zidx",help="Zero subsurface offset index [10]",type=int)
parser.add_argument("-nplots",help="Number of examples to QC [all examples]",type=int)
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-dsize",help="Number of examples in each H5 dataset [20]",type=int)
args = parser.parse_args(remaining_argv)

# Set up IO
sep = seppy.sep(sys.argv)

# Get command line arguments
h5in = args.h5in
verb = sep.yn2zoo(args.verb)
nplots = args.nplots
zidx = args.zidx
dsize = args.dsize

# Open H5 file
hf = h5py.File(h5in,'r')
keys = list(hf.keys())
ntot = int(len(keys)/2)

# Make a random array of ints for plotting
probes = np.unique(np.random.randint(0,ntot,nplots))

movargs = 'gainpanel=a pclip=100'
rhoargs = 'color=j newclip=1 wantscalebar=y'

for ipr in probes:
  # Find the file and the example for the probe
  if(verb): print("Example %d"%(ipr))
  # Choose a random value within the batch
  idx = np.random.randint(1,dsize)
  sep.pltgreymovie(hf[keys[ipr]][idx,:,:,:],greyargs=movargs,o3=0.95,d1=20,d2=20,d3=0.01,bg=True)
  sep.pltgreyimg(hf[keys[ipr+ntot]][idx,:,:],greyargs=rhoargs,d1=20,d2=20,bg=False)

# Close H5 file
hf.close()
