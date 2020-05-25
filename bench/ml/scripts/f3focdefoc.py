"""
Classify focused/defocused faults on F3 dataset

@author: Joseph Jennings
@version: 2020.05.21
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from deeplearn.utils import resizepow2
from deeplearn.keraspredict import focdefocflt
from resfoc.gain import agc
from resfoc.estro import estro_fltfocdefoc, refocusimg
from deeplearn.focuslabels import find_flt_patches
from deeplearn.utils import plotsegprobs, plotseglabel, normalize
from deeplearn.keraspredict import focdefocflt
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
    "fltprbthresh": 0.2,
    "fltnumthresh": 50,
    "mindepth": 1280.0
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
ioArgs.add_argument("-img",help="input image",type=str,required=True)
ioArgs.add_argument("-focwgts",help="input fault focusing CNN weights",type=str,required=True)
ioArgs.add_argument("-focarch",help="input fault focusing CNN architecture",type=str,required=True)
ioArgs.add_argument("-figpfx",help="Output directory of where to save figs",type=str)
# Required arguments
ptchArgs = parser.add_argument_group('Patching parameters')
ptchArgs.add_argument('-ptchx',help='X dimension of patch',type=int,required=True)
ptchArgs.add_argument('-ptchz',help='Z dimension of patch',type=int,required=True)
ptchArgs.add_argument('-strdx',help='X dimension of stride',type=int,required=True)
ptchArgs.add_argument('-strdz',help='Z dimension of stride',type=int,required=True)
# Fault finding arguments
fltArgs = parser.add_argument_group('Fault finding parameters')
fltArgs.add_argument('-fltwgts',help='input fault finding CNN weights',type=str)
fltArgs.add_argument('-fltarch',help="input fault finding CNN architecture",type=str)
fltArgs.add_argument('-fltprbthresh',help='Threshold for fault detection [0.2]',type=float)
fltArgs.add_argument('-fltnumthresh',help='Threshold that patch contains a fault [50]',type=int)
fltArgs.add_argument('-mindepth',help='Minimum depth beyond which begin looking for faults [1200m]',type=float)
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

# Read in the image
iaxes,img = sep.read_file(args.img)
img = img.reshape(iaxes.n,order='F')
img = np.ascontiguousarray(img.T).astype('float32')

[nz,nx,ny] = iaxes.n; [dz,dx,dro] = iaxes.d; [oz,ox,oro] = iaxes.o

mindepth = args.mindepth

# Choose inline
#viewimgframeskey(img,interp='sinc')

iline = resizepow2(img[11]).T


# Read in the fault focusing network
with open(args.focarch,'r') as f:
  focmdl = model_from_json(f.read())
focwgt = focmdl.load_weights(args.focwgts)

if(verb): focmdl.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Read in the fault finding network
fltprb = args.fltprbthresh; fltnum = args.fltnumthresh
with open(args.fltarch,'r') as f:
  fltmdl = model_from_json(f.read())
fltwgt = fltmdl.load_weights(args.fltwgts)
if(verb): fltmdl.summary()

# First find the faults
hasfault,hfimg,timg,pimg = find_flt_patches(iline,fltmdl,dz,mindepth,nzp=args.ptchz,nxp=args.ptchx,
                                            pthresh=fltprb,nthresh=fltnum)

fsize = 15
# Plotting extents
tbeg = 375; tend = 1024
xbeg = 0;   xend = 450
# Amplitude
vmin = np.min(iline); vmax = np.max(iline); pclip = 0.5
plotsegprobs(iline[tbeg:tend,xbeg:xend],pimg[tbeg:tend,xbeg:xend],pmin=fltprb,show=False,vmin=vmin*pclip,vmax=vmax*pclip,
             xmin=xbeg*dx/1000.0,xmax=xend*dx/1000.0,zmin=tbeg*dz,zmax=tend*dz,aratio=1.3,xlabel='X (km)',ylabel='Time (s)')
fig2 = plt.figure(2,figsize=(10,6)); ax2 = fig2.gca()
ax2.imshow(iline[tbeg:tend,xbeg:xend],cmap='gray',interpolation='sinc',vmin=vmin*pclip,vmax=vmax*pclip,
          extent=[xbeg,xend*dx/1000.0,tend*dz,tbeg*dz],aspect=1.2)
ax2.set_xlabel('X (km)',fontsize=fsize)
ax2.set_ylabel('Time (s)',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
plt.show()

# Next predict the fault focusing
focdefoc = focdefocflt(iline,focmdl,rectx=30,rectz=30)

fig4 = plt.figure(4,figsize=(15,8)); ax4 = fig4.gca()
ax4.imshow(iline[tbeg:tend,xbeg:xend],cmap='gray',interpolation='sinc',vmin=vmin*pclip,vmax=vmax*pclip,
           extent=[xbeg,xend*dx/1000.0,tend*dz,tbeg*dz],aspect=1.2)
ax4.imshow(focdefoc[tbeg:tend,xbeg:xend],cmap='jet',alpha=0.1,extent=[xbeg,xend*dx/1000.0,tend*dz,tbeg*dz],aspect=1.2)
ax4.set_xlabel('X (km)',fontsize=fsize)
ax4.set_ylabel('Time (s)',fontsize=fsize)
ax4.tick_params(labelsize=fsize)
plt.show()

cuda.close()
