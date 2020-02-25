"""
Reads in an image or a cube and performs a 2D segmentation
of each image in the cube

@author: Joseph Jennings
@version: 2020.2.23
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from utils.ptyprint import create_inttag
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.utils import plotseglabel, thresh, normalize
from tensorflow.keras.models import model_from_json
import tensorflow as tf
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
ioArgs.add_argument("-imgs",help="input images",type=str,required=True)
ioArgs.add_argument("-wgts",help="input CNN weights",type=str,required=True)
ioArgs.add_argument("-arch",help="input CNN architecture",type=str,required=True)
ioArgs.add_argument("-figpfx",help="Output directory of where to save figs",type=str)
# Required arguments
ptchArgs = parser.add_argument_group('Patching parameters')
ptchArgs.add_argument('-ptchx',help='X dimension of patch',type=int,required=True)
ptchArgs.add_argument('-ptchz',help='Z dimension of patch',type=int,required=True)
ptchArgs.add_argument('-strdx',help='X dimension of stride',type=int,required=True)
ptchArgs.add_argument('-strdz',help='Z dimension of stride',type=int,required=True)
# Optional arguments
parser.add_argument("-thresh",help="Threshold to apply to predictions [0.5]",type=float)
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
parser.add_argument("-show",help="Show plots before saving [n]",type=str)
parser.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
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
gpus  = sep.read_list(args.gpus,[])
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)

# Read in the image
iaxes,imgs = sep.read_file(None,ifname=args.imgs)
imgs = imgs.reshape(iaxes.n,order='F')
if(len(imgs.shape) < 3):
  imgs = np.expand_dims(imgs,axis=-1)
nz = imgs.shape[0]; nx = imgs.shape[1]; nimg = imgs.shape[2]
dz = iaxes.d[0]; dx = iaxes.d[1]

# Perform the patch extraction
nzp = args.ptchz; nxp = args.ptchx
pe = PatchExtractor((nzp,nxp),stride=(args.strdx,args.strdz))

# Read in the network
with open(args.arch,'r') as f:
  model = model_from_json(f.read())

wgts = model.load_weights(args.wgts)

if(verb): model.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Loop over all images
for iimg in range(nimg):
  # Perform the patch extraction
  iptch = pe.extract(imgs[:,:,iimg])
  numpz = iptch.shape[0]; numpx = iptch.shape[1]
  iptch = iptch.reshape([numpx*numpz,nzp,nxp,1])
  # Normalize each patch
  niptch = np.zeros(iptch.shape)
  for ip in range(numpz*numpx):
    niptch[ip,:,:] = normalize(iptch[ip,:,:])
  # Make a prediction
  iprd  = model.predict(niptch,verbose=1)
  # Reconstruct and plot the predictions
  ipra  = iprd.reshape([numpz,numpx,nzp,nxp])
  iprb  = pe.reconstruct(ipra)
  tprb  = thresh(iprb,args.thresh)
  # Plot the prediction and the image
  plotseglabel(normalize(imgs[:,:,iimg]),tprb,color='blue',
             xlabel='X (km)',ylabel='Z (km)',xmin=0.0,xmax=(nx-1)*dx/1000.0,
             zmin=0.0,zmax=(nz-1)*dz/1000.0,vmin=-3.5,vmax=3.5,aratio=1.0,show=show,interp='sinc',
             fname=args.figpfx)

