"""
Reads in residual migration image and detects the faults
within a patch

@author: Joseph Jennings
@version: 2020.05.13
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from tensorflow.keras.models import model_from_json
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.utils import normalize
import tensorflow as tf
from resfoc.gain import agc
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey

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
    "exidx": 600,
    "cropsize": 154,
    "pmin": 0.3,
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
ioArgs.add_argument("-wgts",help="input CNN weights",type=str,required=True)
ioArgs.add_argument("-arch",help="input CNN architecture",type=str,required=True)
ioArgs.add_argument("-figpfx",help="Output directory of where to save figs",type=str)
# Required arguments
ptchArgs = parser.add_argument_group('Patching parameters')
ptchArgs.add_argument('-ptchx',help='X dimension of patch',type=int,required=True)
ptchArgs.add_argument('-ptchz',help='Z dimension of patch',type=int,required=True)
ptchArgs.add_argument('-strdx',help='X dimension of stride',type=int,required=True)
ptchArgs.add_argument('-strdz',help='Z dimension of stride',type=int,required=True)
# Plotting arguments
plotArgs = parser.add_argument_group('Plotting parameters')
plotArgs.add_argument("-aratio",help="Aspect ratio for plotting",type=float)
plotArgs.add_argument("-pmin",help="Minimum probablility to display [0.3]",type=float)
plotArgs.add_argument("-fs",help="First sample for windowing the plots",type=int)
plotArgs.add_argument("-bxidx",help="First X sample for windowing the plots [0]",type=int)
plotArgs.add_argument("-exidx",help="Last X sample for windowing the plots [600]",type=int)
plotArgs.add_argument("-barx",help="X position of colorbar [0.91]",type=float)
plotArgs.add_argument("-barz",help="Z position of colorbar [0.31]",type=float)
plotArgs.add_argument("-hbar",help="Colorbar height [0.37]",type=float)
plotArgs.add_argument("-cropsize",help="Amount to crop on the unsegmented image for removing colorbar",type=int)
# Optional arguments
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
parser.add_argument("-show",help="Show plots before saving [n]",type=str)
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
show = False
if(sep.yn2zoo(args.show)): show = True
gpus  = sep.read_list(args.gpus,[])
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)

# Read in the image
iaxes,img = sep.read_file(args.img)
img = img.reshape(iaxes.n,order='F')
imgt = np.ascontiguousarray(np.transpose(img,(2,0,1))).astype('float32')
[nz,nx,nro] = iaxes.n; [oz,ox,oro] = iaxes.o; [dz,dx,dro] = iaxes.d


# Read in the network
with open(args.arch,'r') as f:
  model = model_from_json(f.read())

wgts = model.load_weights(args.wgts)

if(verb): model.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Extract patches from residually migrated image
per = PatchExtractor((nro,args.ptchz,args.ptchx),stride=(nro,args.strdz,args.strdx))
rptch = np.expand_dims(np.squeeze(per.extract(imgt)),axis=-1)

# Patches from one image
pe = PatchExtractor((args.ptchz,args.ptchx),stride=(args.strdz,args.strdx))
dum = pe.extract(imgt[0])

# Loop over each patch and look probabilities
numpz = rptch.shape[0]; numpx = rptch.shape[1]

out = np.zeros([1,numpz,numpx,nro,args.ptchz,args.ptchx])

rvec = np.zeros(nro)
rhop = np.zeros(dum.shape)

for izp in range(numpz):
  for ixp in range(numpx):
    for iro in range(nro):
      rvec[iro] = model.predict(np.expand_dims(normalize(rptch[izp,ixp,iro,:,:]),axis=0))
      out[0,izp,ixp,iro,:,:] = rvec[iro]
    rhop[izp,ixp,:,:] = np.argmax(rvec)*dro + oro

prds = per.reconstruct(out)
rho  = pe.reconstruct(rhop)

viewimgframeskey(imgt,transp=False,pclip=0.3,show=False,ottl=oro,dttl=dro,ttlstring=r'$\rho$=%f')
viewimgframeskey(prds,transp=False,cmap='jet',show=False,colorbar=True,ottl=oro,dttl=dro,ttlstring=r'$\rho$=%f')

fsize=15
fig2 = plt.figure(3,figsize=(10,6))
ax2 = fig2.gca()
ax2.imshow(agc(imgt[21,:,:]),cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc')
im2 = ax2.imshow(rho,cmap='seismic',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear',alpha=0.1,vmin=0.98,vmax=1.02)
ax2.set_xlabel('X (km)',fontsize=fsize)
ax2.set_ylabel('Z (km)',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
cbar_ax2 = fig2.add_axes([0.91,0.15,0.02,0.70])
cbar2 = fig2.colorbar(im2,cbar_ax2,format='%.2f')
cbar2.solids.set(alpha=1)
cbar2.ax.tick_params(labelsize=fsize)
cbar2.set_label(r'$\rho$',fontsize=fsize)
plt.show()


