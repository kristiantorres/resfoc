"""
Plots the velocity error on top of a migrated image

@author: Joseph Jennings
@version: 2020.04.05
"""
import sys, os, argparse, configparser
import numpy as np
import inpout.seppy as seppy
from resfoc.gain import agc
import matplotlib.pyplot as plt
from utils.image import remove_colorbar

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    "alpha": 0.5,
    "show": 'n',
    "aratio": 1.0,
    "fs": 0,
    "time": "n",
    "km": "y",
    "barx": 0.91,
    "barz": 0.31,
    "hbar": 0.37,
    "exidx": 600,
    "cropsize": 154,
    "thresh": 15,
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
ioArgs.add_argument("-timg",help="input true image",type=str,required=True)
ioArgs.add_argument("-wimg",help="input wrong image",type=str,required=True)
ioArgs.add_argument("-ptb",help="input CNN weights",type=str,required=True)
ioArgs.add_argument("-ivelout",help="Name of output velocity figure",type=str,required=True)
ioArgs.add_argument("-imgout",help="Name of output image figure",type=str,required=True)
# Plotting arguments
plotArgs = parser.add_argument_group('Plotting parameters')
plotArgs.add_argument("-aratio",help="Aspect ratio for plotting",type=float)
plotArgs.add_argument("-fs",help="First sample for windowing the plots",type=int)
plotArgs.add_argument("-bxidx",help="First X sample for windowing the plots [0]",type=int)
plotArgs.add_argument("-exidx",help="Last X sample for windowing the plots [600]",type=int)
plotArgs.add_argument("-time",help="Flag for a time or depth image [n]",type=str)
plotArgs.add_argument("-km",help="Flag for plotting in meters or kilometers [y]",type=str)
plotArgs.add_argument("-barx",help="X position of colorbar [0.91]",type=float)
plotArgs.add_argument("-barz",help="Z position of colorbar [0.31]",type=float)
plotArgs.add_argument("-hbar",help="Colorbar height [0.37]",type=float)
plotArgs.add_argument("-cropsize",help="Amount to crop on the unsegmented image for removing colorbar",type=int)
plotArgs.add_argument("-thresh",help="Threshold for making velocity masks [15 m/s]",type=float)
# Optional arguments
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
parser.add_argument("-show",help="Show plots before saving [n]",type=str)
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
time = False
if(sep.yn2zoo(args.time)): time = True
km = False
if(sep.yn2zoo(args.km)): km = True
thresh = args.thresh

# Read in the image and perturbation
iaxes,img = sep.read_file(args.timg)
img = img.reshape(iaxes.n,order='F')
[nz,nx] = iaxes.n; [dz,dx] = iaxes.d
gimg = agc(img.astype('float32').T).T

waxes,wmg = sep.read_file(args.wimg)
wmg = wmg.reshape(waxes.n,order='F')
gwmg = agc(wmg.astype('float32').T).T

paxes,ptb = sep.read_file(args.ptb)
ptb = ptb.reshape(paxes.n,order='F')

bxidx = 20; exidx = 1000;
fsize = 15; wbox=10; hbox=6
hbar=0.67; wbar=0.02; barz=0.16; barx=0.91
cropsize = 152
# Make the figure of the perturbation (unmasked)
fig1 = plt.figure(figsize=(wbox,hbox))
ax1 = fig1.gca()
im = ax1.imshow(ptb[:,bxidx:exidx],cmap='jet',extent=[bxidx*dx/1000,(exidx-1)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear')
ax1.set_xlabel('X (km)',fontsize=fsize)
ax1.set_ylabel('Z (km)',fontsize=fsize)
ax1.tick_params(labelsize=fsize)
plt.close()

# Plot perturbation on velocity
fig2 = plt.figure(figsize=(wbox,hbox))
ax2 = fig2.gca()
ax2.imshow(gimg[:,bxidx:exidx],vmin=-2.5,vmax=2.5,extent=[bxidx*dx/1000,(exidx-1)*dx/1000.0,nz*dz/1000.0,0.0],cmap='gray',interpolation='sinc')
mask1 = np.ma.masked_where((ptb) < thresh, ptb)
mask2 = np.ma.masked_where((ptb) > -thresh, ptb)
ax2.imshow(mask1[:,bxidx:exidx],extent=[bxidx*dx/1000,(exidx-1)*dx/1000.0,nz*dz/1000.0,0.0],alpha=0.3,
    cmap='jet',vmin=-100,vmax=100,interpolation='bilinear')
ax2.imshow(mask2[:,bxidx:exidx],extent=[bxidx*dx/1000,(exidx-1)*dx/1000.0,nz*dz/1000.0,0.0],alpha=0.3,
    cmap='jet',vmin=-100,vmax=100,interpolation='bilinear')
ax2.set_xlabel('X (km)',fontsize=fsize)
ax2.set_ylabel('Z (km)',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
# Colorbar
cbar_ax = fig2.add_axes([barx,barz,wbar,hbar])
cbar = fig2.colorbar(im,cbar_ax,format='%.0f',boundaries=np.arange(-100,101,1))
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig(args.ivelout,bbox_inches='tight',transparent=True,dpi=150)
if(show):
  plt.show()
plt.close()

fig3 = plt.figure(figsize=(wbox,hbox))
ax3 = fig3.gca()
ax3.imshow(gimg[:,bxidx:exidx],vmin=-2.5,vmax=2.5,extent=[bxidx*dx/1000,(exidx-1)*dx/1000.0,nz*dz/1000.0,0.0],cmap='gray',interpolation='sinc')
ax3.set_xlabel('X (km)',fontsize=fsize)
ax3.set_ylabel('Z (km)',fontsize=fsize)
ax3.tick_params(labelsize=fsize)
# Colorbar
cbar_ax = fig3.add_axes([barx,barz,wbar,hbar])
cbar = fig3.colorbar(im,cbar_ax,format='%.0f',boundaries=np.arange(-100,101,1))
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig(args.imgout+'-tmp.png',bbox_inches='tight',transparent=True,dpi=150)
plt.close()
remove_colorbar(args.imgout+'-tmp.png',cropsize=cropsize,opath=args.imgout+'.png')


fig4 = plt.figure(figsize=(wbox,hbox))
ax4 = fig4.gca()
ax4.imshow(gwmg[:,bxidx:exidx],vmin=-2.5,vmax=2.5,extent=[bxidx*dx/1000,(exidx-1)*dx/1000.0,nz*dz/1000.0,0.0],cmap='gray',interpolation='sinc')
ax4.set_xlabel('X (km)',fontsize=fsize)
ax4.set_ylabel('Z (km)',fontsize=fsize)
ax4.tick_params(labelsize=fsize)
# Colorbar
cbar_ax = fig4.add_axes([barx,barz,wbar,hbar])
cbar = fig4.colorbar(im,cbar_ax,format='%.0f',boundaries=np.arange(-100,101,1))
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label('Velocity (km/s)',fontsize=fsize)
plt.savefig(args.wimgout+'-tmp.png',bbox_inches='tight',transparent=True,dpi=150)
plt.close()
remove_colorbar(args.wimgout+'-tmp.png',cropsize=cropsize,opath=args.wimgout+'.png')

