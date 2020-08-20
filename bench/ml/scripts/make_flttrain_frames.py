"""
Makes a movie of the training process by showing the predictions
on a test dataset

@author: Joseph Jennings
@version: 22.02.2020
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import glob
import h5py
from deeplearn.dataloader import load_allflddata
from deeplearn.utils import thresh, plotseglabel
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from genutils.ptyprint import create_inttag, progressbar
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    'nepochs': None,
    'fs': 200,
    'show': 'n'
    }
if args.conf_file:
  config = configparser.ConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

# Set defaults
parser.set_defaults(**defaults)

# Input files
ioArgs = parser.add_argument_group('Inputs and outputs')
ioArgs.add_argument("-prddir",help="Prediction directory",type=str)
ioArgs.add_argument("-flddat",help="Field data for predictions",type=str)
ioArgs.add_argument("-figdir",help="Directory where to store figures",type=str)
# Required arguments
requiredNamed = parser.add_argument_group('required parameters')
requiredNamed.add_argument('-xlidx',help='Crossline index on which prediction was made',type=int,required=True)
requiredNamed.add_argument('-nxo',help='X dimension of entire output image',type=int,required=True)
requiredNamed.add_argument('-nzo',help='Z dimension of entire output image',type=int,required=True)
requiredNamed.add_argument('-ptchx',help='X dimension of patch',type=int,required=True)
requiredNamed.add_argument('-ptchz',help='Z dimension of patch',type=int,required=True)
requiredNamed.add_argument('-strdx',help='X dimension of stride',type=int,required=True)
requiredNamed.add_argument('-strdz',help='Z dimension of stride',type=int,required=True)
requiredNamed.add_argument('-nptchx',help='Number of patches per image in X dimension',type=int,required=True)
requiredNamed.add_argument('-nptchz',help='Number of patches per image in Z dimension',type=int,required=True)
# Optional arguments
parser.add_argument('-fs',help="First sample for windowing the time axis [200]",type=int)
parser.add_argument('-thresh',help='Threshold to apply to predictions [0.2]',type=float)
parser.add_argument('-nepochs',help="Number of epochs to plot [all]",type=int)
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
parser.add_argument("-show",help="Show the figures before saving [n]",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
show = False
if(sep.yn2zoo(args.show)): show = True

# Get the predictions from each epoch
allpreds = sorted(glob.glob(args.prddir + '/' + '*.h5'))

# Build the patch extractor
nzp = args.ptchz; nxp = args.ptchx
nzo = args.nzo;   nxo = args.nxo
pe = PatchExtractor((nzp,nxp),stride=(args.strdx,args.strdz))
dptch = pe.extract(np.zeros([nzo,nxo]))

# Load in the test data
numpx = args.nptchx; numpz = args.nptchz
xlidx = args.xlidx; nptch = numpx*numpz
flddat = load_allflddata(args.flddat,nptch)

# Create the image from the test data
iimg = flddat[xlidx*nptch:(xlidx+1)*nptch,:,:]
iimg = iimg.reshape([numpz,numpx,nzp,nxp])
rimg = pe.reconstruct(iimg)

# Loop over all predictions
if(args.nepochs == None):
  nepochs = len(allpreds)
else:
  nepochs = args.nepochs
fs = args.fs
for iep in progressbar(range(nepochs), 'nepochs'):
  with h5py.File(allpreds[iep],'r') as hf:
    pred = hf['pred'][:]
  # Reconstruct and threshold the predictions
  iprd = pred[xlidx*nptch:(xlidx+1)*nptch,:,:]
  iprd = iprd.reshape([numpz,numpx,nzp,nxp])
  rprd = pe.reconstruct(iprd)
  tprd = thresh(rprd,args.thresh)
  # Plot the label and the image
  plotseglabel(rimg[fs:,:],tprd[fs:,:],color='blue',
             xlabel='Xline (km)',ylabel='Time (s)',xmin=0.0,xmax=(nxo-1)*25/1000.0,
             zmin=(fs-1)*0.004,zmax=(nzo-1)*0.004,vmin=-2.5,vmax=2.5,aratio=6.5,show=show,
             interp='sinc',fname=args.figdir + '/ep%d'%(iep))

