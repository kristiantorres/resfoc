"""
Segment faults on some migrated images

@author: Joseph Jennings
@version: 2020.06.25
"""
import sys, os, glob, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from deeplearn.utils import resizepow2
from deeplearn.keraspredict import focdefocflt
from resfoc.gain import agc
from deeplearn.utils import plotsegprobs, plotseglabel, normalize
from deeplearn.keraspredict import segmentfaults
from utils.image import remove_colorbar
from utils.movie import viewimgframeskey
from numba import cuda
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "y",
    "gpus": [],
    "show": "n",
    "thresh": 0.5,
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
ioArgs.add_argument("-imgdir",help="Directory containing images",type=str,required=True)
ioArgs.add_argument("-figpfx",help="Output directory of where to save figs",type=str)
# Required arguments
ptchArgs = parser.add_argument_group('Patching parameters')
ptchArgs.add_argument('-ptchx',help='X dimension of patch',type=int,required=True)
ptchArgs.add_argument('-ptchz',help='Z dimension of patch',type=int,required=True)
ptchArgs.add_argument('-strdx',help='X dimension of stride',type=int,required=True)
ptchArgs.add_argument('-strdz',help='Z dimension of stride',type=int,required=True)
# Neural network arguments
fltArgs = parser.add_argument_group('Fault finding parameters')
fltArgs.add_argument('-wgts',help='Input CNN weights',type=str)
fltArgs.add_argument('-arch',help="Input CNN architecture",type=str)
fltArgs.add_argument('-thresh',help='Threshold for fault probability [0.2]',type=float)
# Other arguments
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-show",help="Flag for showing plots ([y] or n)",type=str)
parser.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep()

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
gpus  = sep.read_list(args.gpus,[])
show  = sep.yn2zoo(args.show)
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)

# Choose inline
#viewimgframeskey(img,interp='sinc')

# Read in the fault focusing network
with open(args.arch,'r') as f:
  mdl = model_from_json(f.read())
mdl.load_weights(args.wgts)

if(verb): mdl.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Read in the images
imgs = glob.glob(args.imgdir + '/*.H')
titles = ['Focused','Defocused','Muted-CNN','Muted-Semblance','Semblance','CNN']
names = ['foc','def','mutcnn','mutsmb','smb','cnn']
k = 0
for iimg in imgs:
  iaxes,img = sep.read_file(iimg)
  img = img.reshape(iaxes.n,order='F')
  img = np.ascontiguousarray(img.T).astype('float32')

  [nz,nx] = iaxes.n; [dz,dx] = iaxes.d; [oz,ox] = iaxes.o

  imgg = agc(img).T

  begz = 130; endz = 400
  begx = 50;  endx = 462
  pred = segmentfaults(imgg,mdl,nzp=args.ptchz,nxp=args.ptchx)
  #fig = plt.figure(figsize=(8,8)); ax = fig.gca()
  #ax.imshow(imgg[begz:endz,begx:endx],cmap='gray',interpolation='sinc',vmin=-3.0,vmax=3.0)
  #ax.set_xlabel('X',fontsize=18)
  #ax.set_ylabel('Z',fontsize=18)
  #ax.tick_params(labelsize=18)
  plotsegprobs(imgg[begz:endz,begx:endx],pred[begz:endz,begx:endx],args.thresh,show=False,
               vmin=-3.0,vmax=3.0,hbox=8,wbox=8,title=titles[k],fname='./fig/mymigsold/'+names[k],barz=0.2,hbar=0.6)
  k += 1

