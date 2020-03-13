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
from deeplearn.utils import plotseglabel, thresh, normalize, resample
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
    "aratio": 1.0,
    "fs": 0,
    "time": "n"
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
ptchArgs.add_argument('-nxo',help='Output x for resampling before patching [1024]',type=int)
ptchArgs.add_argument('-nzo',help='Output z for resampling before patching [512]',type=int)
ptchArgs.add_argument('-ptchx',help='X dimension of patch',type=int,required=True)
ptchArgs.add_argument('-ptchz',help='Z dimension of patch',type=int,required=True)
ptchArgs.add_argument('-strdx',help='X dimension of stride',type=int,required=True)
ptchArgs.add_argument('-strdz',help='Z dimension of stride',type=int,required=True)
# Optional arguments
parser.add_argument("-thresh",help="Threshold to apply to predictions [0.5]",type=float)
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
parser.add_argument("-show",help="Show plots before saving [n]",type=str)
parser.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
parser.add_argument("-aratio",help="Aspect ratio for plotting",type=float)
parser.add_argument("-fs",help="First sample for windowing the plots",type=int)
parser.add_argument("-time",help="Flag for a time or depth image [n]",type=str)
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
time = False
if(sep.yn2zoo(args.time)): time = True
gpus  = sep.read_list(args.gpus,[])
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)

# Read in the image
iaxes,imgs = sep.read_file(None,ifname=args.imgs)
imgs = imgs.reshape(iaxes.n,order='F')
if(len(imgs.shape) < 3):
  imgs = np.expand_dims(imgs,axis=-1)
else:
  imgs = np.transpose(imgs,(2,0,1))
nz = imgs.shape[0]; nx = imgs.shape[1]; nimg = imgs.shape[2]
dz = iaxes.d[0]; dx = iaxes.d[1]

# Perform the patch extraction
nzo = args.nzo;   nxo = args.nxo
nzp = args.ptchz; nxp = args.ptchx
pe = PatchExtractor((nzp,nxp),stride=(args.strdx,args.strdz))

## Read in the network
with open(args.arch,'r') as f:
  model = model_from_json(f.read())

wgts = model.load_weights(args.wgts)

if(verb): model.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Loop over all images
for iimg in range(nimg):
  # Resample the images to the output size
  if(nimg == 1):
    rimg,ds = resample(imgs[:,:,0],[nzo,nxo],kind='linear',ds=[dz,dx])
  else:
    rimg,ds = resample(imgs[iimg,:,:],[nzo,nxo],kind='linear',ds=[dz,dx])
  # Perform the patch extraction
  iptch = pe.extract(rimg)
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
  if(not time):
    dz /= 1000
  plotseglabel(normalize(rimg)[args.fs:,:],tprb[args.fs:,:],color='blue',
             xlabel='X (km)',ylabel='Z (km)',xmin=0.0,xmax=(nx-1)*ds[1]/1000.0,
             zmin=args.fs*dz,zmax=(nz-1)*dz,vmin=-2.5,vmax=2.5,aratio=args.aratio,show=show,interp='sinc',
             fname=args.figpfx)

