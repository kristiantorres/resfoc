"""
Creates a well-focused fault image. Generated image
contains highly folded and faulted structures

@author: Joseph Jennings
@version: 2020.06.03
"""
import sys, os, argparse, configparser
import numpy as np
import inpout.seppy as seppy
from velocity.stdmodels import velfaultsrandom
from deeplearn.utils import resample,thresh
from scaas.trismooth import smooth
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
import matplotlib.pyplot as plt
from utils.plot import plot_imgvelptb
from utils.signal import ampspec2d
from utils.movie import viewimgframeskey

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "fx": 373,
    "nxw": 512,
    "na": 64,
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
ioArgs.add_argument("-vel",help="Output velocity model",type=str,required=True)
ioArgs.add_argument("-ref",help="Output reflectivity",type=str,required=True)
ioArgs.add_argument("-lbl",help="Output label",type=str,required=True)
ioArgs.add_argument("-cnv",help="Focused convolution image",type=str,required=True)
ioArgs.add_argument("-img",help="Focused migrated image",type=str,required=True)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Create IO for writing at the end
sep = seppy.sep()

# Create layered model
nx = 1024; nz = 512
vel,ref,cnv,lbl = velfaultsrandom(nz=nz,nx=nx)
dx = 10; dz = 10

plt.figure(1)
plt.imshow(vel,cmap='jet',extent=[0,1024*0.01,512*0.01,0.0])
plt.figure(2)
plt.imshow(lbl,cmap='jet',extent=[0,1024*0.01,512*0.01,0.0])
plt.figure(3)
plt.imshow(ref,cmap='gray',extent=[0,1024*0.01,512*0.01,0.0])
plt.figure(4)
plt.imshow(cnv,cmap='gray',extent=[0,1024*0.01,512*0.01,0.0])
plt.show()

# Create migration velocity
velmig = smooth(vel,rect1=30,rect2=30)

# Acquisition geometry
dsx = 20; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=52,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(velmig,cmap='jet',show=True)

# Create data axes
ntu = 6500; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)

# Model true linearized data
dtd = 0.004
allshot = prp.model_lindata(velmig,ref,wav,dtd,verb=True,nthrds=24)

# Taper for migration
prp.build_taper(70,150)

# Wave equation depth migration (both with anomaly)
img = prp.wem(velmig,allshot,wav,dtd,lap=True,verb=True,nthrds=24)

# Output inputs
sep.write_file(args.vel,vel,ds=[dz,dx])
sep.write_file(args.ref,ref,ds=[dz,dx])
sep.write_file(args.lbl,lbl,ds=[dz,dx])
sep.write_file(args.cnv,cnv,ds=[dz,dx])
sep.write_file(args.img,img,ds=[dz,dx])

