"""
Tries to classify faults a focused or defocused

@author: Joseph Jennings
@version: 2020.05.17
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import h5py
import tensorflow as tf
import random
from tensorflow.keras.models import model_from_json
from deeplearn.keraspredict import focdefocflt
from resfoc.gain import agc
from resfoc.estro import estro_fltfocdefoc
from utils.image import remove_colorbar
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
ioArgs.add_argument("-img",help="input images",type=str,required=True)
ioArgs.add_argument("-wgts",help="input CNN weights",type=str,required=True)
ioArgs.add_argument("-arch",help="input CNN architecture",type=str,required=True)
ioArgs.add_argument("-figpfx",help="Output directory of where to save figs",type=str)
# Required arguments
ptchArgs = parser.add_argument_group('Patching parameters')
ptchArgs.add_argument('-ptchx',help='X dimension of patch',type=int,required=True)
ptchArgs.add_argument('-ptchz',help='Z dimension of patch',type=int,required=True)
ptchArgs.add_argument('-strdx',help='X dimension of stride',type=int,required=True)
ptchArgs.add_argument('-strdz',help='Z dimension of stride',type=int,required=True)
# Other arguments
parser.add_argument("-thresh",help="Threshold to apply to predictions [0.5]",type=float)
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
gimg = agc(img)
gimgt = np.ascontiguousarray(np.transpose(gimg,(0,2,1)))

[nz,nx,nro] = iaxes.n; [dz,dx,dro] = iaxes.d; [oz,ox,oro] = iaxes.o

## Read in the network
with open(args.arch,'r') as f:
  model = model_from_json(f.read())

wgts = model.load_weights(args.wgts)

#if(verb): model.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Plot the prediction for rho=1
#focmapro1 = focdefocflt(gimgt[21],model,nzp=args.ptchz,nxp=args.ptchx,rectx=30,rectz=30)
fsize = 16
#fig3 = plt.figure(3,figsize=(8,8)); ax3 = fig3.gca()
#ax3.imshow(gimgt[21],cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
#im3 = ax3.imshow(focmapro1,cmap='jet',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear',vmin=0.0,vmax=1.0,alpha=0.1)
#ax3.set_xlabel('X (km)',fontsize=fsize)
#ax3.set_ylabel('Z (km)',fontsize=fsize)
#ax3.tick_params(labelsize=fsize)
#cbar_ax3 = fig3.add_axes([0.91,0.115,0.02,0.76])
#cbar3 = fig3.colorbar(im3,cbar_ax3,format='%.1f')
#cbar3.solids.set(alpha=1)
#cbar3.ax.tick_params(labelsize=fsize)
#cbar3.set_label('Probability',fontsize=fsize)
#plt.savefig('./fig/focdefocprbrho1.png',bbox_inches='tight',transparent=True,dpi=150)
#plt.close()
#
## Plot the defocused image
#fig2 = plt.figure(2,figsize=(8,8)); ax2 = fig2.gca()
#ax2.imshow(gimgt[21],cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
#ax2.set_xlabel('X (km)',fontsize=fsize)
#ax2.set_ylabel('Z (km)',fontsize=fsize)
#ax2.tick_params(labelsize=fsize)
#cbar_ax2 = fig2.add_axes([0.91,0.115,0.02,0.76])
#cbar2 = fig3.colorbar(im3,cbar_ax2,format='%.1f')
#cbar2.solids.set(alpha=1)
#cbar2.ax.tick_params(labelsize=fsize)
#cbar2.set_label('Probability',fontsize=fsize)
#plt.savefig('./fig/justdefoc-tmp.png',dpi=150,transparent=True,bbox_inches='tight')
#plt.close()
#remove_colorbar('./fig/justdefoc-tmp.png',160,opath='./fig/justdefoc.png')

focmap = np.zeros(gimgt.shape)
for iro in range(nro):
  focmap[iro] = focdefocflt(gimgt[iro],model,nzp=args.ptchz,nxp=args.ptchx,rectx=30,rectz=30)
  rho = oro + iro*dro
  if(args.show):
    fig1 = plt.figure(1,figsize=(8,8)); ax1 = fig1.gca()
    ax1.imshow(gimgt[iro],cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
    im1 = ax1.imshow(focmap[iro],cmap='jet',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear',vmin=0.0,vmax=1.0,alpha=0.1)
    ax1.set_title(r'$\rho$=%.4f'%(rho),fontsize=fsize)
    ax1.set_xlabel('X (km)',fontsize=fsize)
    ax1.set_ylabel('Z (km)',fontsize=fsize)
    ax1.tick_params(labelsize=fsize)
    cbar_ax1 = fig1.add_axes([0.91,0.115,0.02,0.76])
    cbar1 = fig1.colorbar(im1,cbar_ax1,format='%.1f')
    cbar1.solids.set(alpha=1)
    cbar1.ax.tick_params(labelsize=fsize)
    cbar1.set_label('Probability',fontsize=fsize)
    plt.savefig('./fig/prbrho%d'%(iro),dpi=150,transparent=True,bbox_inches='tight')
    plt.close()

