"""
Creates focused fault patch training data
for fault segmentation

@author: Joseph Jennings
@version: 2020.06.22
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import glob
import h5py
from utils.ptyprint import progressbar, create_inttag
import numpy as np
from resfoc.gain import agc
from deeplearn.faultlabels import faultseg_labels
from deeplearn.utils import plotseglabel, resample, thresh
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "datdir": "/data/sep/joseph29/projects/resfoc/bench/dat/focdefoc",
    "iprefix": "fog-",
    "lprefix": "lbl-",
    "out": "",
    "ptchz": 128,
    "ptchx": 128,
    "strdz": 64,
    "strdx": 64,
    "norm": 'y',
    "rectx": 3,
    "rectz": 3,
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
ioArgs.add_argument("-datdir",help="Directory containingfocused  images",type=str)
ioArgs.add_argument("-cprefix",help="Prefix to convolution image",type=str)
ioArgs.add_argument("-mprefix",help="Prefix to focused subsurface offset gathers",type=str)
ioArgs.add_argument("-lprefix",help="Prefix to fault labels",type=str)
ioArgs.add_argument("-migout",help="Output migrated image H5 file",type=str)
ioArgs.add_argument("-cnvout",help="Output convolution image H5 file",type=str)
# Patching arguments
ptchArgs = parser.add_argument_group('Patching parameters')
ptchArgs.add_argument("-ptchz",help="Size of patch in z [128]",type=int)
ptchArgs.add_argument("-ptchx",help="Size of patch in x [128]",type=int)
ptchArgs.add_argument("-strdz",help="Patch stride (overlap) in z  [64]",type=int)
ptchArgs.add_argument("-strdx",help="Patch stride (overlap) in x [64]",type=int)
# Label arguments
lblArgs = parser.add_argument_group('Label creation parameters')
lblArgs.add_argument("-norm",help="Normalize each patch from -1 to 1 [y]",type=str)
lblArgs.add_argument("-rectx",help="Label smoothing in the x direction [3 points]",type=int)
lblArgs.add_argument("-rectz",help="Label smoothing in the z direction [3 points]",type=int)
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
strdz = args.strdz; strdx = args.strdx

# Windowing parameters
fx = args.fx; nxw = args.nxw
fz = args.fz; nzw = args.nzw

# Flags
norm   = sep.yn2zoo(args.norm)
verb   = sep.yn2zoo(args.verb)
qcplot = sep.yn2zoo(args.qcplot)

# Get SEPlib files
cnvfiles = sorted(glob.glob(args.datdir + '/' + args.cprefix + '*.H'))
migfiles = sorted(glob.glob(args.datdir + '/' + args.mprefix + '*.H'))
lblfiles = sorted(glob.glob(args.datdir + '/' + args.lprefix + '*.H'))
ntot = len(cnvfiles)

# Create H5 File
hfm = h5py.File(args.migout,'w')
hfc = h5py.File(args.cnvout,'w')

for iex in progressbar(range(ntot), "nfiles:"):
  # Read in and window the migrated image
  maxes,mimg = sep.read_file(migfiles[iex])
  mimg = mimg.reshape(maxes.n,order='F')
  mimg = mimg.T
  zoimgt = agc(mimg[16]).T
  zoimg = zoimgt[fz:fz+nzw,fx:fx+nxw]
  # Read in and window the convolution image
  caxes,cimg = sep.read_file(cnvfiles[iex])
  cimg = cimg.reshape(caxes.n,order='F')
  # Read in the label
  laxes,lbl = sep.read_file(lblfiles[iex])
  lbl = lbl.reshape(laxes.n,order='F')
  lblr = thresh(resample(lbl,[512,1024],kind='linear'),0)
  lblw = lblr[fz:fz+nzw,fx:fx+nxw]
  # Make the patch labels
  mptch,lwptch,limg = faultseg_labels(zoimg,lblw,nxp=nxp,nzp=nzp,strdz=args.strdz,strdx=args.strdx,
                                rectx=args.rectx,rectz=args.rectz,lblimg=True)
  cptch,lrptch      = faultseg_labels(cimg ,lblr,nxp=nxp,nzp=nzp,strdz=args.strdz,strdx=args.strdx,
                                rectx=args.rectx,rectz=args.rectz)
  mnptch = mptch.shape[0]; cnptch = cptch.shape[0]
  # Write the data to output H5 file
  datatag = create_inttag(iex,ntot)
  # Write convolution images
  hfm.create_dataset("x"+datatag, (mnptch,nzp,nxp,1), data=np.expand_dims(mptch ,axis=-1), dtype=np.float32)
  hfm.create_dataset("y"+datatag, (mnptch,nzp,nxp,1), data=np.expand_dims(lwptch,axis=-1), dtype=np.float32)
  # Write migration images
  hfc.create_dataset("x"+datatag, (cnptch,nzp,nxp,1), data=np.expand_dims(cptch ,axis=-1), dtype=np.float32)
  hfc.create_dataset("y"+datatag, (cnptch,nzp,nxp,1), data=np.expand_dims(lrptch,axis=-1), dtype=np.float32)
  # Plot image and segmentation label
  if(qcplot):
    pclip = 0.3
    plotseglabel(zoimg,limg,pclip=pclip,show=False)
    plotseglabel(cnvw ,limg,pclip=pclip,show=True)

# Close the H5 file
hfm.close()
hfc.close()

