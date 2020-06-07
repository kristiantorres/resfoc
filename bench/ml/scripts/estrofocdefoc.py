"""
Estimates rho from defocused faults

@author: Joseph Jennings
@version: 2020.05.20
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
from resfoc.estro import estro_fltfocdefoc, refocusimg
from deeplearn.focuslabels import find_flt_patches
from deeplearn.utils import plotsegprobs, plotseglabel, thresh
from utils.image import remove_colorbar
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
ioArgs.add_argument("-img",help="input images",type=str,required=True)
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
gimg = agc(img)
gimgt = np.ascontiguousarray(np.transpose(gimg,(0,2,1)))

[nz,nx,nro] = iaxes.n; [dz,dx,dro] = iaxes.d; [oz,ox,oro] = iaxes.o

mindepth = args.mindepth

# Read in the fault focusing network
with open(args.focarch,'r') as f:
  focmdl = model_from_json(f.read())
focwgt = focmdl.load_weights(args.focwgts)

if(verb): focmdl.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Set GPU memory here
#tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05)

# Defocused image
rho1img = gimgt[21]

# Read in the fault detection network
if(args.fltarch is ''):
  onlyfaults = False
  rho,fltfocs = estro_fltfocdefoc(gimgt,focmdl,dro,oro)
else:
  fltprb = args.fltprbthresh; fltnum = args.fltnumthresh
  with open(args.fltarch,'r') as f:
    fltmdl = model_from_json(f.read())
  fltwgt = fltmdl.load_weights(args.fltwgts)
  if(verb): fltmdl.summary()
  # First find the faults
  hasfault,hfimg,timg,pimg = find_flt_patches(rho1img,fltmdl,dz,mindepth,pthresh=fltprb,nthresh=fltnum)
  plotsegprobs(rho1img,pimg,pmin=fltprb,show=False)
  plotseglabel(rho1img,timg,color='blue',show=False)
  plt.figure(3)
  plt.imshow(rho1img,cmap='gray',interpolation='sinc')
  plt.imshow(thresh(hfimg,0.0),cmap='jet',alpha=0.1)
  plt.show()
  rho,fltfocs = estro_fltfocdefoc(gimgt,focmdl,dro,oro,hasfault=hasfault)

cuda.close()

# Read in rho
raxes,rhosmb = sep.read_file('../focdat/dat/refocus/mltest/mltestdogrhomask.H')
rhosmb = rhosmb.reshape(raxes.n,order='F')

# Refocus the image with both
rfismb = refocusimg(gimgt,rhosmb,dro)
rfiflt = refocusimg(gimgt,rho,dro)

faxes,fog = sep.read_file('../focdat/dat/focdefoc/mltestfog.H')
fog = np.ascontiguousarray(fog.reshape(faxes.n,order='F').T)
gfog = agc(fog.astype('float32'))
fog = gfog[16,256:512+256,:]

# Write out the refocused images
sep.write_file('rfismbcompmask.H',rfismb,ds=[dx,dz])
sep.write_file('rfifltfoc.H',rfiflt,ds=[dx,dz])

# Plot rho on the defocused image
fsize=15
fig2 = plt.figure(1,figsize=(8,8)); ax2 = fig2.gca()
ax2.imshow(rho1img,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
im2 = ax2.imshow(rhosmb.T,cmap='seismic',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear',vmin=0.98,vmax=1.02,alpha=0.1)
ax2.set_xlabel('X (km)',fontsize=fsize)
ax2.set_ylabel('Z (km)',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
ax2.set_title(r"Semblance",fontsize=fsize)
cbar_ax2 = fig2.add_axes([0.91,0.15,0.02,0.70])
cbar2 = fig2.colorbar(im2,cbar_ax2,format='%.2f')
cbar2.solids.set(alpha=1)
cbar2.ax.tick_params(labelsize=fsize)
cbar2.set_label(r'$\rho$',fontsize=fsize)
plt.savefig('./fig/rhosembmaskimg.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

fig3 = plt.figure(2,figsize=(8,8)); ax3 = fig3.gca()
ax3.imshow(rho1img,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
im3 = ax3.imshow(rho,cmap='seismic',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear',vmin=0.98,vmax=1.02,alpha=0.1)
ax3.set_xlabel('X (km)',fontsize=fsize)
ax3.set_ylabel('Z (km)',fontsize=fsize)
ax3.tick_params(labelsize=fsize)
ax3.set_title(r"Fault focusing",fontsize=fsize)
cbar_ax3 = fig3.add_axes([0.91,0.15,0.02,0.70])
cbar3 = fig3.colorbar(im3,cbar_ax3,format='%.2f')
cbar3.solids.set(alpha=1)
cbar3.ax.tick_params(labelsize=fsize)
cbar3.set_label(r'$\rho$',fontsize=fsize)
plt.savefig('./fig/rhofltimg.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

# Plot the refocused images
fig4 = plt.figure(3,figsize=(8,8)); ax4 = fig4.gca()
ax4.imshow(rfismb,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax4.set_xlabel('X (km)',fontsize=fsize)
ax4.set_ylabel('Z (km)',fontsize=fsize)
ax4.set_title('Semblance',fontsize=fsize)
ax4.tick_params(labelsize=fsize)
plt.savefig('./fig/rfisembmask.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

fig5 = plt.figure(4,figsize=(8,8)); ax5 = fig5.gca()
ax5.imshow(rfiflt,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax5.set_xlabel('X (km)',fontsize=fsize)
ax5.set_ylabel('Z (km)',fontsize=fsize)
ax5.set_title('Fault focus',fontsize=fsize)
ax5.tick_params(labelsize=fsize)
plt.savefig('./fig/rfiflt.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

fig6 = plt.figure(5,figsize=(8,8)); ax6 = fig6.gca()
ax6.imshow(fog.T,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax6.set_xlabel('X (km)',fontsize=fsize)
ax6.set_ylabel('Z (km)',fontsize=fsize)
ax6.set_title('Well focused',fontsize=fsize)
ax6.tick_params(labelsize=fsize)
plt.savefig('./fig/wellfoc.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()
