"""
Selects defocused patches from non-stationary defocused images

@author: Joseph Jennings
@version: 2020.05.17
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import h5py
from utils.ptyprint import progressbar, create_inttag
import numpy as np
from resfoc.gain import agc
from deeplearn.dataloader import load_all_unlabeled_data,load_unlabeled_flat_data
from deeplearn.focuslabels import extract_defocpatches
from deeplearn.utils import plotsegprob
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "focptch":,
    "defptch":,
    "lblptch":,
    "fpbptch":,
    "dpbptch":,
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
ioArgs.add_argument("-focptch",help="H5 file containing focused patches",type=str,required=True)
ioArgs.add_argument("-defptch",help="H5 file containing defocused patches",type=str,required=True)
ioArgs.add_argument("-lblptch",help="H5 file containing fault labels",type=str,required=True)
ioArgs.add_argument("-fpbptch",help="H5 file containing focused probabilities",type=str,required=True)
ioArgs.add_argument("-dpbptch",help="H5 file containing defocused probabilities",type=str,required=True)
ioArgs.add_argument("-out",help="Output H5 file",type=str)
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

# Read in the H5 files
focdat = load_all_unlabeled_data(args.focptch)
defdat = load_all_unlabeled_data(args.defptch)
fltdat = load_all_unlabeled_data(args.lblptch)
focprb = load_unlabeled_flat_data()
defprb = load_unlabeled_flat_data()

# QC the fault predictions

## First get all of the defocused images
#defocs = []
#for iex in progressbar(range(ntot), "nfiles:"):
#  # Read in focused image
#  faxes,fimg = sep.read_file(focfiles[iex])
#  fimg = fimg.reshape(faxes.n,order='F')
#  fimg = np.transpose(fimg,(2,0,1))
#  zofimg = fimg[16,fz:fz+nzw,fx:fx+nxw]
#  zofimgt = np.ascontiguousarray(zofimg.T)
#  gzofimg = agc(zofimgt.astype('float32')).T
#  # Read in the defocused image
#  daxes,dimg = sep.read_file(deffiles[iex])
#  dimg = dimg.reshape(daxes.n,order='F')
#  dimg = np.transpose(dimg,(2,0,1))
#  zodimg = dimg[16,fz:fz+nzw,fx:fx+nxw]
#  zodimgt = np.ascontiguousarray(zodimg.T)
#  gzodimg = agc(zodimgt.astype('float32')).T
#  # Read in the label
#  laxes,lbl = sep.read_file(lblfiles[iex])
#  lbl = lbl.reshape(laxes.n,order='F')
#  lblw = lbl[fz:fz+nzw,fx:fx+nxw]
#  # Make the patch labels
#  idefoc,dptch,fptch,nrm = extract_defocpatches(gzodimg,gzofimg,lblw,                                       # Inputs
#                                          nxp=args.ptchx,nzp=args.ptchz,strdx=args.strdx,strdz=args.strdz,  # Patching
#                                          pixthresh=args.pthresh,metric=args.metric,focthresh=args.fthresh, # Thresholding
#                                          imgs=True,qcptchgrd=False)                                        # Flags
#  # Flatten and QC the patches
#  if(qcplot):
#    viewimgframeskey(idefoc,transp=False,interp='sinc',vmin=-2.5,vmax=2.5,show=True)
#  # Append to output defocused images
#  defocs.append(np.asarray(idefoc))
#
## Convert to numpy array
#defocs = np.concatenate(defocs, axis=0)
## Write the data to output H5 file
#ndefoc = defocs.shape[0]
#for idfc in progressbar(range(ndefoc), "ndefoc:"):
#  datatag = create_inttag(idfc,ndefoc)
#  hf.create_dataset("x"+datatag, (nzp,nxp,1), data=np.expand_dims(defocs[idfc],axis=-1), dtype=np.float32)
#
## Close the H5 file
#hf.close()

