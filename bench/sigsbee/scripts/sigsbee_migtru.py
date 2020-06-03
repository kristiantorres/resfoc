"""
Migrates the sigsbee model with the correct velocity

@author: Joseph Jennings
@version: 2020.06.01
"""
import sys, os, argparse, configparser
import numpy as np
import inpout.seppy as seppy
from deeplearn.utils import resample
from scaas.velocity import create_randomptbs_loc
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
ioArgs.add_argument("-vel",help="Input migration velocity model",type=str,required=True)
ioArgs.add_argument("-dat",help="Input data",type=str,required=True)
ioArgs.add_argument("-sbeg",help="Beginning shot index",type=int,required=True)
ioArgs.add_argument("-send",help="Beginning shot index",type=int,required=True)
ioArgs.add_argument("-fimgo",help="Focused migrated image (offset)",type=str,required=True)
# Optional arguments
parser.add_argument("-nthrds",help="Number of OMP threads [5]",type=int)
parser.add_argument("-fx",help="First x sample for window [373]",type=int)
parser.add_argument("-nxw",help="Size of window [512]",type=int)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Create IO for writing at the end
sep = seppy.sep()

# First read in sigsbee model
vaxes,vel = sep.read_file(args.vel)
vel = np.ascontiguousarray(vel.reshape(vaxes.n,order='F')).astype('float32')
daxes,dat = sep.read_file(args.dat)
dat = dat.reshape(daxes.n,order='F')
dat = np.ascontiguousarray(np.transpose(dat,(2,0,1))).astype('float32')
sbeg = args.sbeg; send = args.send
datw = dat[sbeg:send,:,:]

# Get axes
nz,nx = vaxes.n; dz,dx = vaxes.d
ntd,nrx,nsx = daxes.n; dtd,dx,dsx = daxes.d
oz,ox,osx = daxes.o

# Acquisition geometry
bx = 50; bz = 50
srange = send-sbeg
# Create all shots and select from them
sxs = np.linspace(osx,(nsx-1)*dsx,nsx).astype('int')
orange = sxs[sbeg]
print("Imaging shots %d - %d"%(sbeg,send-1))
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=srange,osx=orange,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(vel,cmap='jet',show=True)

# Create data axes
ntu = 10000; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)

# Taper for migration
prp.build_taper(70,150)

prp.plot_taper(vel)

# Wave equation depth migration (both with anomaly)
img = prp.wem(vel,datw,wav,dtd,nh=16,lap=True,verb=True,nthrds=args.nthrds)
nh,oh,dh = prp.get_off_axis()

# Window to the region of interest
nxw = args.nxw; nw = args.fx
imgw = img[:,:,nw:nw+nxw]

# Window and transpose the image and the label
imgwt = np.transpose(imgw,(0,2,1)) # [nh,nz,nx] -> [nh,nx,nz]

# Output images
sep.write_file(args.fimgo,imgwt.T,ds=[dz,dx,dh],os=[0,0,oh])

