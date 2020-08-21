"""
Creates defocused patches from training images

@author: Joseph Jennings
@version: 2020.05.12
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import glob
import h5py
from genutils.ptyprint import progressbar, create_inttag
import numpy as np
from resfoc.gain import agc
from deeplearn.focuslabels import extract_defocpatches
from deeplearn.utils import plotseglabel
from genutils.movie import viewimgframeskey
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "datdir": "/data/sep/joseph29/projects/resfoc/bench/dat/focdefoc",
    "fprefix": "fog-",
    "dprefix": "dog-",
    "lprefix": "lbl-",
    "ptchz": 64,
    "ptchx": 64,
    "strdz": 32,
    "strdx": 32,
    "norm": 'y',
    "pthresh": 30,
    "fthresh": 0.65,
    "metric": 'corr',
    "nedefoc": 50000,
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
ioArgs = parser.add_argument_group('Input/Output')
ioArgs.add_argument("-datdir",help="Directory containing focused images",type=str)
ioArgs.add_argument("-fprefix",help="Prefix to focused subsurface offset gathers",type=str)
ioArgs.add_argument("-dprefix",help="Prefix to defocused subsurface offset gathers",type=str)
ioArgs.add_argument("-lprefix",help="Prefix to fault labels",type=str)
ioArgs.add_argument("-filebeg",help="Beginning file index for training data",type=int)
ioArgs.add_argument("-fileend",help="Ending file index for training data",type=int)
ioArgs.add_argument("-out",help="Output H5 file",type=str)
# Training data arguments
trnArgs = parser.add_argument_group('Training data parameters')
# Patching arguments
ptchArgs = parser.add_argument_group('Patching parameters')
ptchArgs.add_argument("-ptchz",help="Size of patch in z [128]",type=int)
ptchArgs.add_argument("-ptchx",help="Size of patch in x [128]",type=int)
ptchArgs.add_argument("-strdz",help="Patch stride (overlap) in z  [64]",type=int)
ptchArgs.add_argument("-strdx",help="Patch stride (overlap) in x [64]",type=int)
# Label arguments
lblArgs = parser.add_argument_group('Label creation parameters')
lblArgs.add_argument("-pthresh",help="Number of fault pixels in a patch required to make a label [20]",type=int)
lblArgs.add_argument("-norm",help="Normalize each patch from -1 to 1 [y]",type=str)
lblArgs.add_argument("-metric",help="Metric to use for comparing focused and defocused image ([mse] or ssim)",type=str)
lblArgs.add_argument("-fthresh",help="Threshold to determine if image is focused or not [0.5]",type=float)
# Window arguments
windArgs = parser.add_argument_group('Windowing parameters')
windArgs.add_argument("-fx",help="First x sample [256]",type=int)
windArgs.add_argument("-nxw",help="Length of window in x [512]",type=int)
windArgs.add_argument("-fz",help="First z sample [138]",type=int)
windArgs.add_argument("-nzw",help="Length of window in z [256]",type=int)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
othArgs.add_argument("-qcplot",help="Plot the labels and images (y or [n])",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Setup IO
sep = seppy.sep()

# Get command line parameters
nzp = args.ptchz; nxp = args.ptchx

# Windowing parameters
fx = args.fx; nxw = args.nxw
fz = args.fz; nzw = args.nzw

# Flags
norm   = sep.yn2zoo(args.norm)
verb   = sep.yn2zoo(args.verb)
qcplot = sep.yn2zoo(args.qcplot)

# Get SEPlib files
focfiles = sorted(glob.glob(args.datdir + '/' + args.fprefix + '*.H'))[args.filebeg:args.fileend]
deffiles = sorted(glob.glob(args.datdir + '/' + args.dprefix + '*.H'))[args.filebeg:args.fileend]
lblfiles = sorted(glob.glob(args.datdir + '/' + args.lprefix + '*.H'))[args.filebeg:args.fileend]
ntot = len(focfiles)

# Create H5 File
hf = h5py.File(args.out,'w')

# First get all of the defocused images
defocs = []
for iex in progressbar(range(ntot), "nfiles:"):
  # Read in focused image
  faxes,fimg = sep.read_file(focfiles[iex])
  fimg = fimg.reshape(faxes.n,order='F')
  fimg = np.transpose(fimg,(2,0,1))
  zofimg = fimg[16,fz:fz+nzw,fx:fx+nxw]
  zofimgt = np.ascontiguousarray(zofimg.T)
  gzofimg = agc(zofimgt.astype('float32')).T
  # Read in the defocused image
  daxes,dimg = sep.read_file(deffiles[iex])
  dimg = dimg.reshape(daxes.n,order='F')
  dimg = np.transpose(dimg,(2,0,1))
  zodimg = dimg[16,fz:fz+nzw,fx:fx+nxw]
  zodimgt = np.ascontiguousarray(zodimg.T)
  gzodimg = agc(zodimgt.astype('float32')).T
  # Read in the label
  laxes,lbl = sep.read_file(lblfiles[iex])
  lbl = lbl.reshape(laxes.n,order='F')
  lblw = lbl[fz:fz+nzw,fx:fx+nxw]
  # Make the patch labels
  idefoc,dptch,fptch,nrm = extract_defocpatches(gzodimg,gzofimg,lblw,                                       # Inputs
                                          nxp=args.ptchx,nzp=args.ptchz,strdx=args.strdx,strdz=args.strdz,  # Patching
                                          pixthresh=args.pthresh,metric=args.metric,focthresh=args.fthresh, # Thresholding
                                          imgs=True,qcptchgrd=False)                                        # Flags
  # Flatten and QC the patches
  if(qcplot):
    viewimgframeskey(idefoc,transp=False,interp='sinc',vmin=-2.5,vmax=2.5,show=True)
  # Append to output defocused images
  defocs.append(np.asarray(idefoc))


# Convert to numpy array
defocs = np.concatenate(defocs, axis=0)
# Write the data to output H5 file
ndefoc = defocs.shape[0]
for idfc in progressbar(range(ndefoc), "ndefoc:"):
  datatag = create_inttag(idfc,ndefoc)
  hf.create_dataset("x"+datatag, (nzp,nxp,1), data=np.expand_dims(defocs[idfc],axis=-1), dtype=np.float32)

# Close the H5 file
hf.close()

