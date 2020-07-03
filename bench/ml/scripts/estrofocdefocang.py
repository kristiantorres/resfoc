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
from deeplearn.keraspredict import focdefocang
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
[nz,na,nx,nro] = iaxes.n; [dz,da,dx,dro] = iaxes.d; [oz,oa,ox,oro] = iaxes.o
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

# Predict focusing on rho=1 image
focangro1 = focdefocang(gimgt[20],focmdl)

sep.write_file('focprbrho1full.H',focangro1,ds=[dz,dx])

# Write out the estimated rho
sep.write_file('rhoangcnn3.H',rho,ds=[dz,dz])

cuda.close()

# WIndowing parameters
fx =  49; nx = 400
fz = 120; nz = 300

# Defocused image
rho1img = stkgt[21]

rho1imgw = rho1img[fz:fz+nz,fx:fx+nx]

# Read in rho
raxes,rhosmb = sep.read_file('../focdat/dat/refocus/mltest/mltestdogrhomask2.H')
rhosmb = rhosmb.reshape(raxes.n,order='F')
rhosmb = np.ascontiguousarray(rhosmb).astype('float32')

raxes2,rhosmb2 = sep.read_file('../focdat/dat/refocus/mltest/mltestdogrho.H')
rhosmb2 = rhosmb2.reshape(raxes2.n,order='F')
rhosmb2 = np.ascontiguousarray(rhosmb2).astype('float32')

# Window the rhos
rhosmbw  = rhosmb[fz:fz+nz,fx:fx+nx]
rhosmb2w = rhosmb2[fz:fz+nz,fx:fx+nx]
rhow     = rho[fz:fz+nz,fx:fx+nx]

## Refocus the image with both
rfismb  = refocusimg(stkgt,rhosmb,dro)
rfismb2 = refocusimg(stkgt,rhosmb2,dro)
rficnn  = refocusimg(stkgt,rho,dro)

# Window the images
rfismbw  = rfismb[fz:fz+nz,fx:fx+nx]
rfismb2w = rfismb2[fz:fz+nz,fx:fx+nx]
rficnnw  = rficnn[fz:fz+nz,fx:fx+nx]

faxes,fog = sep.read_file('../focdat/dat/focdefoc/mltestfog.H')
fog = np.ascontiguousarray(fog.reshape(faxes.n,order='F').T)
gfog = agc(fog.astype('float32'))
fog = gfog[16,305:305+nx,fz:fz+nz]

# Write out the focusing probabilities
sep.write_file('focprb.H',fltfocs.T,os=[0.0,0.0,oro],ds=[dz,dx,dro])

# Write out the refocused images
sep.write_file('rfismbcomp2.H',rfismb,ds=[dx,dz])
sep.write_file('rficnnfoc2.H',rficnn,ds=[dx,dz])

# Plotting window
fxi = 305; fzi = 120
dx /= 1000.0; dz /= 1000.0

# Plot rho on the defocused image
fsize=16
fig2 = plt.figure(1,figsize=(10,10)); ax2 = fig2.gca()
ax2.imshow(rho1imgw,cmap='gray',extent=[fxi*dx,(fxi+nx)*dx,(fzi+nz)*dz,fzi*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
im2 = ax2.imshow(rhosmbw,cmap='seismic',extent=[fxi*dx,(fxi+nx)*dx,(fzi+nz)*dz,fzi*dz],interpolation='bilinear',vmin=0.975,vmax=1.025,alpha=0.2)
ax2.set_xlabel('X (km)',fontsize=fsize)
ax2.set_ylabel('Z (km)',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
#ax2.set_title(r"Semblance",fontsize=fsize)
cbar_ax2 = fig2.add_axes([0.925,0.205,0.02,0.58])
cbar2 = fig2.colorbar(im2,cbar_ax2,format='%.2f')
cbar2.solids.set(alpha=1)
cbar2.ax.tick_params(labelsize=fsize)
cbar2.set_label(r'$\rho$',fontsize=fsize)
plt.savefig('./fig/videofigs/rhosembimg.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

fig2 = plt.figure(1,figsize=(10,10)); ax2 = fig2.gca()
ax2.imshow(rho1imgw,cmap='gray',extent=[fxi*dx,(fxi+nx)*dx,(fzi+nz)*dz,fzi*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
im2 = ax2.imshow(rhosmb2w,cmap='seismic',extent=[fxi*dx,(fxi+nx)*dx,(fzi+nz)*dz,fzi*dz],interpolation='bilinear',vmin=0.975,vmax=1.025,alpha=0.2)
ax2.set_xlabel('X (km)',fontsize=fsize)
ax2.set_ylabel('Z (km)',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
#ax2.set_title(r"Semblance",fontsize=fsize)
cbar_ax2 = fig2.add_axes([0.925,0.205,0.02,0.58])
cbar2 = fig2.colorbar(im2,cbar_ax2,format='%.2f')
cbar2.solids.set(alpha=1)
cbar2.ax.tick_params(labelsize=fsize)
cbar2.set_label(r'$\rho$',fontsize=fsize)
plt.savefig('./fig/videofigs/rhosembimgfull.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

fig3 = plt.figure(2,figsize=(10,10)); ax3 = fig3.gca()
ax3.imshow(rho1imgw,cmap='gray',extent=[fxi*dx,(fxi+nx)*dx,(fzi+nz)*dz,fzi*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
im3 = ax3.imshow(rhow,cmap='seismic',extent=[fxi*dx,(fxi+nx)*dx,(fzi+nz)*dz,fzi*dz],interpolation='bilinear',vmin=0.975,vmax=1.025,alpha=0.2)
ax3.set_xlabel('X (km)',fontsize=fsize)
ax3.set_ylabel('Z (km)',fontsize=fsize)
ax3.tick_params(labelsize=fsize)
#ax3.set_title(r"Angle focusing",fontsize=fsize)
cbar_ax3 = fig3.add_axes([0.925,0.205,0.02,0.58])
cbar3 = fig3.colorbar(im3,cbar_ax3,format='%.2f')
cbar3.solids.set(alpha=1)
cbar3.ax.tick_params(labelsize=fsize)
cbar3.set_label(r'$\rho$',fontsize=fsize)
plt.savefig('./fig/videofigs/rhoangimg.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

# Plot the refocused images
fig4 = plt.figure(3,figsize=(10,10)); ax4 = fig4.gca()
ax4.imshow(rfismbw,cmap='gray',extent=[fxi*dx,(fxi+nx)*dx,(fzi+nz)*dz,fzi*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax4.set_xlabel('X (km)',fontsize=fsize)
ax4.set_ylabel('Z (km)',fontsize=fsize)
#ax4.set_title('Semblance',fontsize=fsize)
ax4.tick_params(labelsize=fsize)
plt.savefig('./fig/videofigs/rfisemb.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

fig4 = plt.figure(3,figsize=(10,10)); ax4 = fig4.gca()
ax4.imshow(rfismb2w,cmap='gray',extent=[fxi*dx,(fxi+nx)*dx,(fzi+nz)*dz,fzi*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax4.set_xlabel('X (km)',fontsize=fsize)
ax4.set_ylabel('Z (km)',fontsize=fsize)
#ax4.set_title('Semblance',fontsize=fsize)
ax4.tick_params(labelsize=fsize)
plt.savefig('./fig/videofigs/rfisembfull.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

fig5 = plt.figure(4,figsize=(10,10)); ax5 = fig5.gca()
ax5.imshow(rficnnw,cmap='gray',extent=[fxi*dx,(fxi+nx)*dx,(fzi+nz)*dz,fzi*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax5.set_xlabel('X (km)',fontsize=fsize)
ax5.set_ylabel('Z (km)',fontsize=fsize)
#ax5.set_title('Angle focus',fontsize=fsize)
ax5.tick_params(labelsize=fsize)
plt.savefig('./fig/videofigs/rfiang.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

fig6 = plt.figure(5,figsize=(10,10)); ax6 = fig6.gca()
ax6.imshow(fog.T,cmap='gray',extent=[fxi*dx,(fxi+nx)*dx,(fzi+nz)*dz,fzi*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax6.set_xlabel('X (km)',fontsize=fsize)
ax6.set_ylabel('Z (km)',fontsize=fsize)
#ax6.set_title('Well focused',fontsize=fsize)
ax6.tick_params(labelsize=fsize)
plt.savefig('./fig/videofigs/wellfoc.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

fig7 = plt.figure(6,figsize=(10,10)); ax7 = fig7.gca()
ax7.imshow(rho1imgw,cmap='gray',extent=[fxi*dx,(fxi+nx)*dx,(fzi+nz)*dz,fzi*dz],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax7.set_xlabel('X (km)',fontsize=fsize)
ax7.set_ylabel('Z (km)',fontsize=fsize)
ax7.tick_params(labelsize=fsize)
plt.savefig('./fig/videofigs/defocstk.png',transparent=True,dpi=150,bbox_inches='tight')
plt.close()

plt.show()
