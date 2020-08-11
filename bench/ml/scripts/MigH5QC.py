"""
Performs a quality-check on the poststack (RTM) image
training data. Using Vplot, plots random examples and
prints the associated file name.

Because this program uses Vplot, you must have
SEPlib correctly installed and configured
to run

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
    "nplots": 2,
    "ro1idx": 5,
    "zidx": 10
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
parser.add_argument("-h5in",help="Input h5 file",required=True,type=str)
# Optional arguments
parser.add_argument("-ro1idx",help="Index of rho=1 image [5]",type=int)
parser.add_argument("-zidx",help="Zero subsurface offset index [10]",type=int)
parser.add_argument("-nplots",help="Number of examples to QC [all examples]",type=int)
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
args = parser.parse_args(remaining_argv)

# Set up IO
sep = seppy.sep(sys.argv)

# Get command line arguments
sepdir = args.sepdir
h5in = args.h5in
verb = sep.yn2zoo(args.verb)
nplots = args.nplots
zidx = args.zidx
ro1idx = args.ro1idx

# First get total number of examples
imgfiles = sorted(glob.glob(sepdir + '/img*.H'))
lblfiles = sorted(glob.glob(sepdir + '/lbl*.H'))

assert(len(imgfiles) == len(lblfiles)), "Must have as many features as labels. Exiting"

# Get number of examples
ntot = 0; filect = {}
for ifile in zip(imgfiles,lblfiles):
  filect[ntot] = ifile
  ntot += (sep.read_header(None,ifname=ifile[0])).n[4]

ncts = np.asarray(sorted(filect.keys()))

# Read in H5 file
with h5py.File(h5in,'r') as hf:
  data   = hf['data'][:]
  labels = hf['labels'][:]

# Make a random array of ints for plotting
probes = np.unique(np.random.randint(0,ntot,nplots))

gryargs = 'pclip=100'
rhoargs = 'color=j newclip=1 wantscalebar=y'

for ipr in probes:
  # Find the file and the example for the probe
  if(verb): print("Example %d"%(ipr))
  dist = np.abs(ncts - ipr)
  idx = dist.argmin()
  if(ipr < ncts[idx]):
    idx -= 1
  key = ncts[idx]
  iex = ipr-key
  if(verb): print("Files: %s %s, example %d in file"%(filect[key][0],filect[key][1],iex))
  # Read in the probed example from the SEP file
  iaxes,img = sep.read_file(None,ifname=filect[key][0])
  img = img.reshape(iaxes.n,order='F')
  raxes,rho = sep.read_file(None,ifname=filect[key][1])
  rho = rho.reshape(raxes.n,order='F')
  # Plot the probed example from H5 file
  sep.pltgreyimg(data[ipr,:,:],greyargs=gryargs,d1=20,d2=20,bg=True)
  sep.pltgreyimg(labels[ipr,:,:],greyargs=rhoargs,d1=20,d2=20,bg=False)
  # Plot the probed example from the SEP files
  imgdiff = img[:,:,zidx,ro1idx,iex] - data[ipr,:,:]
  print("Img diff: max=%f min=%f"%(np.max(imgdiff),np.min(imgdiff)))
  rhodiff = rho[:,:,iex] - labels[ipr,:,:]
  print("Rho diff: max=%f min=%f"%(np.max(rhodiff),np.min(rhodiff)))

