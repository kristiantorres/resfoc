"""
Creates fault segmentation training data

@author: Joseph Jennings
@version: 2020.05.16
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import glob
import h5py
from genutils.ptyprint import progressbar, create_inttag
import numpy as np
from deeplearn.utils import plotseglabel, normalize
from deeplearn.faultlabels import cleanfaultseg_labels
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
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
ioArgs.add_argument("-lprefix",help="Prefix to fault labels",type=str)
ioArgs.add_argument("-out",help="Output H5 file",type=str)
ioArgs.add_argument("-filebeg",help="Beginning file to use",type=int)
ioArgs.add_argument("-fileend",help="Ending file to use",type=int)
# Patching arguments
ptchArgs = parser.add_argument_group('Patching parameters')
ptchArgs.add_argument("-ptchz",help="Size of patch in z [128]",type=int)
ptchArgs.add_argument("-ptchx",help="Size of patch in x [128]",type=int)
ptchArgs.add_argument("-strdz",help="Patch stride (overlap) in z  [64]",type=int)
ptchArgs.add_argument("-strdx",help="Patch stride (overlap) in x [64]",type=int)
# Label arguments
lblArgs = parser.add_argument_group('Label creation parameters')
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
focfiles = sorted(glob.glob(args.datdir + '/' + args.iprefix + '*.H'))[args.filebeg:args.fileend]
lblfiles = sorted(glob.glob(args.datdir + '/' + args.lprefix + '*.H'))[args.filebeg:args.fileend]
ntot = len(focfiles)

# Create PatchExtractor
nzp = args.ptchz; nxp = args.ptchx
pshape = (nzp,nxp)
pstride = (args.strdz,args.strdx)
pe = PatchExtractor(pshape,stride=pstride)

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
  # Extract patches
  iptch = pe.extract(zoimg)
  lptch = pe.extract(lblw)
  # Flatten
  pz = iptch.shape[0]; px = iptch.shape[1]
  iptch = iptch.reshape([pz*px,nzp,nxp])
  lptch = lptch.reshape([pz*px,nzp,nxp])
  # Process patches
  #cleanfaultseg_labels(iptch,lptch)
  niptch = np.zeros(iptch.shape)
  # Normalize the images
  for ip in range(pz*px):
    niptch[ip,:,:] = normalize(iptch[ip,:,:])
  # Write the data to output H5 file
  datatag = create_inttag(iex,ntot)
  hf.create_dataset("x"+datatag, (pz*px,nzp,nxp,1), data=np.expand_dims(niptch,axis=-1), dtype=np.float32)
  hf.create_dataset("y"+datatag, (pz*px,nzp,nxp,1), data=np.expand_dims(lptch,axis=-1), dtype=np.float32)
  # Plot image and segmentation label
  if(qcplot):
    pclip = 0.5
    fig = plt.figure(figsize=(8,6)); ax = fig.gca()
    ax.imshow(zoimg,cmap='gray',interpolation='sinc',vmin=np.min(zoimg)*pclip,vmax=np.max(zoimg)*pclip)
    ax.tick_params(labelsize=14)
    plotseglabel(zoimg,lblw,pclip=pclip,show=True)

# Close the H5 file
hf.close()

