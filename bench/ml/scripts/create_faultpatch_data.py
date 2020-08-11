"""
Creates fault patch training data

@author: Joseph Jennings
@version: 2020.05.12
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import glob
import h5py
from utils.ptyprint import progressbar, create_inttag
import numpy as np
from deeplearn.focuslabels import faultpatch_labels
from deeplearn.utils import plotseglabel
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
    "ptchz": 64,
    "ptchx": 64,
    "strdz": 32,
    "strdx": 32,
    "norm": 'y',
    "pthresh": 20,
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
ioArgs.add_argument("-iprefix",help="Prefix to focused subsurface offset gathers",type=str)
ioArgs.add_argument("-lprefix",help="Prefix to focused subsurface offset gathers",type=str)
ioArgs.add_argument("-out",help="Output H5 file",type=str)
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
focfiles = sorted(glob.glob(args.datdir + '/' + args.iprefix + '*.H'))
lblfiles = sorted(glob.glob(args.datdir + '/' + args.lprefix + '*.H'))
ntot = len(focfiles)

# Create H5 File
hf = h5py.File(args.out,'w')

for iex in progressbar(range(ntot), "nfiles:"):
  # Read in and window the focused image
  faxes,fimg = sep.read_file(focfiles[iex])
  fimg = fimg.reshape(faxes.n,order='F')
  fimg = np.transpose(fimg,(2,0,1))
  zoimg = fimg[16,fz:fz+nzw,fx:fx+nxw]
  # Read in the label
  laxes,lbl = sep.read_file(lblfiles[iex])
  lbl = lbl.reshape(laxes.n,order='F')
  lblw = lbl[fz:fz+nzw,fx:fx+nxw]
  # Make the patch labels
  iptch,lptch,limg = faultpatch_labels(zoimg,lblw,strdx=args.strdx,strdz=args.strdz,
                                       pixthresh=args.pthresh,ptchimg=True,qcptchgrd=False)
  # Flatten and QC the patches
  pz = lptch.shape[0]; px = lptch.shape[1]
  lptchf = lptch.reshape([pz*px,nzp,nxp])
  iptchf = iptch.reshape([pz*px,nzp,nxp])
  # Write the data to output H5 file
  datatag = create_inttag(iex,ntot)
  hf.create_dataset("x"+datatag, (pz*px,nzp,nxp,1), data=np.expand_dims(iptchf,axis=-1), dtype=np.float32)
  hf.create_dataset("y"+datatag, (pz*px,1), data=lptchf[:,int(nzp/2),int(nxp/2)], dtype=np.float32)
  # Plot image and segmentation label
  if(qcplot):
    pclip = 0.3
    plotseglabel(zoimg,lblw,pclip=pclip,show=False)
    # Plot the patch label on top of the image
    plt.figure(2,figsize=(10,6))
    plt.imshow(zoimg,cmap='gray',vmin=pclip*np.min(zoimg),vmax=pclip*np.max(zoimg),interpolation='sinc')
    plt.imshow(limg,cmap='seismic',interpolation='bilinear',vmin=0,vmax=1.0,alpha=0.1)
    plt.show()

# Close the H5 file
hf.close()

