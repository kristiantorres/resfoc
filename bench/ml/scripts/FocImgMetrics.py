"""
Computes Metrics for the focused image classification

@author: Joseph Jennings
@version: 2020.04.14
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import h5py
import numpy as np
from utils.ptyprint import create_inttag, progressbar
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.utils import plotseglabel, plotsegprobs, thresh, normalize, resample
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from resfoc.gain import agc
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    "thresh": 0.5,
    "show": 'n',
    "gpus": [],
    "aratio": 1.0,
    "fs": 0,
    "time": "n",
    "km": "y",
    "barx": 0.91,
    "barz": 0.31,
    "hbar": 0.37,
    "xidx": 600,
    "cropsize": 154,
    "qc": "n",
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
ioArgs.add_argument("-focimgs",help="input focused images",type=str,required=True)
ioArgs.add_argument("-cnvimgs",help="input convolution images",type=str,required=True)
ioArgs.add_argument("-fltlbls",help="Fault labels",type=str,required=True)
ioArgs.add_argument("-wgts",help="input CNN weights",type=str,required=True)
ioArgs.add_argument("-arch",help="input CNN architecture",type=str,required=True)
# Required arguments
ptchArgs = parser.add_argument_group('Patching parameters')
ptchArgs.add_argument('-nxo',help='Total image size',type=int,required=True)
ptchArgs.add_argument('-nzo',help='Total image size',type=int,required=True)
ptchArgs.add_argument('-ptchx',help='X dimension of patch',type=int,required=True)
ptchArgs.add_argument('-ptchz',help='Z dimension of patch',type=int,required=True)
ptchArgs.add_argument('-strdx',help='X dimension of stride',type=int,required=True)
ptchArgs.add_argument('-strdz',help='Z dimension of stride',type=int,required=True)
# Optional arguments
parser.add_argument("-qc",help="Visual QC of the predictions (y or [n])",type=str)
parser.add_argument("-thresh",help="Threshold to apply to predictions [0.5]",type=float)
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
parser.add_argument("-show",help="Show plots before saving [n]",type=str)
parser.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# SEP IO
sep = seppy.sep()

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
qc    = sep.yn2zoo(args.qc)
gpus  = sep.read_list(args.gpus,[])
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)

# Read in the network
with open(args.arch,'r') as f:
  model = model_from_json(f.read())

wgts = model.load_weights(args.wgts)

if(verb): model.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Read in the H5 datasets
hff = h5py.File(args.focimgs,'r')
hfc = h5py.File(args.cnvimgs,'r')
hfl = h5py.File(args.fltlbls,'r')

# Read in each example
fkeys = list(hff.keys())
ckeys = list(hfc.keys())
lkeys = list(hfl.keys())

# Number of examples
nex = len(fkeys)

# Sampling for plotting
dz = 0.01; dx = 0.01

# Create the patch extractor and the output array
nzp = args.ptchz; nxp = args.ptchx
nzo = args.nzo;   nxo = args.nxo
pe = PatchExtractor((nzp,nxp),stride=(args.strdz,args.strdx))
dptch = pe.extract(np.zeros([nzo,nxo]))
numpz = dptch.shape[0]; numpx = dptch.shape[1]

# Output predictions
fpreds = np.zeros([nex,numpx*numpz,nxp,nzp,1])
cpreds = np.zeros([nex,numpx*numpz,nxp,nzp,1])

# Output metrics
ioufs = np.zeros(nex); ioucs = np.zeros(nex)

for iex in progressbar(range(nex), "iex:"):
  # Get the example
  ifoc = np.asarray(hff[fkeys[iex]]).T
  icnv = np.asarray(hfc[ckeys[iex]]).T
  ilbl = np.asarray(hfl[lkeys[iex]]).T
  # Perform the patch extraction
  ifp = pe.extract(ifoc)
  icp = pe.extract(icnv)
  ilp = pe.extract(ilbl)
  ifp = ifp.reshape([numpx*numpz,nzp,nxp,1])
  icp = icp.reshape([numpx*numpz,nzp,nxp,1])
  ilp = ilp.reshape([numpx*numpz,nzp,nxp,1])
  # Normalize each patch
  nifp = np.zeros(ifp.shape)
  nicp = np.zeros(icp.shape)
  for ip in range(numpz*numpx):
    nifp[ip,:,:] = normalize(ifp[ip,:,:])
    nicp[ip,:,:] = normalize(icp[ip,:,:])
  # Make a prediction
  ifprd  = model.predict(nifp,verbose=0)
  icprd  = model.predict(nicp,verbose=0)
  # Save the predictions to an array
  fpreds[iex] = ifprd
  cpreds[iex] = icprd
  # Threshold the predictions
  ifprdt = thresh(ifprd,args.thresh)
  icprdt = thresh(icprd,args.thresh)
  # Reconstruct the images
  ifpre = pe.reconstruct(ifprd.reshape([numpz,numpx,nzp,nxp]))
  icpre = pe.reconstruct(icprd.reshape([numpz,numpx,nzp,nxp]))
  # Evaluate the prediction
  ioufs[iex] = jaccard_score(ifprdt.flatten(),ilp.flatten())
  ioucs[iex] = jaccard_score(icprdt.flatten(),ilp.flatten())
  # Plot the prediction
  if(qc):
    gifoc = agc(ifoc.astype('float32').T).T
    plotsegprobs(gifoc,ifpre,
               xlabel='X (km)',ylabel='Z (km)',xmin=0.0,xmax=(nxo)*dx,
               zmin=0,zmax=(nzo)*dz,vmin=-2.5,vmax=2.5,show=False,interp='sinc',
               pmin=0.3,alpha=0.7,ticksize=14,barlabelsize=14,barx=0.91,
               hbar=0.67,wbox=10,labelsize=14,barz=0.16)
    plotsegprobs(icnv,icpre,
               xlabel='X (km)',ylabel='Z (km)',xmin=0.0,xmax=(nxo)*dx,
               zmin=0,zmax=(nzo)*dz,vmin=-2.5,vmax=2.5,show=True,interp='sinc',
               pmin=0.3,alpha=0.7,ticksize=14,barlabelsize=14,barx=0.91,
               hbar=0.51,wbox=8,labelsize=14,barz=0.24,fname='./fig/cnvpred760')

print("Mean Focused IOU: %f"%(np.mean(ioufs)))
print("Mean Convolution IOU: %f"%(np.mean(ioucs)))

# Close the H5 files
hff.close(); hfc.close(); hfl.close()

