"""
Estimates rho from defocused faults and angle gathers

@author: Joseph Jennings
@version: 2020.05.27
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
from joblib import Parallel, delayed
from resfoc.estro import estro_fltangfocdefoc, refocusimg
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
img = np.ascontiguousarray(img.T).astype('float32') # [nro,nx,na,nz]
stk = np.sum(img,axis=2)
## Apply AGC
# Angle gathers
[nz,na,nx,nro] = iaxes.n; [dz,da,dx,dro] = iaxes.d; [oz,da,ox,oro] = iaxes.o
gimg =  np.asarray(Parallel(n_jobs=24)(delayed(agc)(img[iro]) for iro in range(nro)))
gimgt = np.ascontiguousarray(np.transpose(gimg,(0,2,3,1))) # [nro,nx,na,nz] -> [nro,na,nz,nx]
# Stack
stkg = agc(stk)
stkgt = np.transpose(stkg,(0,2,1)) # [nro,nx,nz] -> [nro,nz,nx]

# Read in the fault focusing network
with open(args.focarch,'r') as f:
  focmdl = model_from_json(f.read())
focwgt = focmdl.load_weights(args.focwgts)

if(verb): focmdl.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Read in the fault detection network
rho,fltfocs = estro_fltangfocdefoc(gimgt,focmdl,dro,oro,rectz=40,rectx=40)

# Write out the estimated rho
sep.write_file('rhoangcnn.H',rho,ds=[dz,dz])

cuda.close()

# Defocused image
rho1img = stkgt[21]

# Read in rho
raxes,rhosmb = sep.read_file('../focdat/refocus/mltest/mltestdogrho.H')
rhosmb = rhosmb.reshape(raxes.n,order='F')

## Refocus the image with both
rfismb = refocusimg(stkgt,rhosmb,dro)
rficnn = refocusimg(stkgt,rho,dro)

faxes,fog = sep.read_file('../focdat/focdefoc/mltestfog.H')
fog = np.ascontiguousarray(fog.reshape(faxes.n,order='F').T)
gfog = agc(fog.astype('float32'))
fog = gfog[16,256:512+256,:]

# Write out the refocused images
#sep.write_file('rfismbcomp.H',rfismb,ds=[dx,dz])
sep.write_file('rficnnfoc.H',rficnn,ds=[dx,dz])

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

# Plot the refocused images
fig4 = plt.figure(3,figsize=(8,8)); ax4 = fig4.gca()
ax4.imshow(rfismb,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax4.set_xlabel('X (km)',fontsize=fsize)
ax4.set_ylabel('Z (km)',fontsize=fsize)
ax4.set_title('Semblance',fontsize=fsize)
ax4.tick_params(labelsize=fsize)

fig5 = plt.figure(4,figsize=(8,8)); ax5 = fig5.gca()
ax5.imshow(rficnn,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax5.set_xlabel('X (km)',fontsize=fsize)
ax5.set_ylabel('Z (km)',fontsize=fsize)
ax5.set_title('Fault focus',fontsize=fsize)
ax5.tick_params(labelsize=fsize)

fig6 = plt.figure(5,figsize=(8,8)); ax6 = fig6.gca()
ax6.imshow(fog.T,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax6.set_xlabel('X (km)',fontsize=fsize)
ax6.set_ylabel('Z (km)',fontsize=fsize)
ax6.set_title('Well focused',fontsize=fsize)
ax6.tick_params(labelsize=fsize)

plt.show()
