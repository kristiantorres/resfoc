"""
Creates a well focused image given a velocity
anomaly and a velocity and reflectivity model

@author: Joseph Jennings
@version: 2020.05.03
"""
import sys, os, argparse, configparser
import numpy as np
import inpout.seppy as seppy
from scaas.trismooth import smooth
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from utils.plot import plot_imgvelptb
import matplotlib.pyplot as plt

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
ioArgs.add_argument("-vel",help="Input velocity model",type=str,required=True)
ioArgs.add_argument("-ref",help="Input reflectivity",type=str,required=True)
ioArgs.add_argument("-ptb",help="Input velocity anomaly",type=str,required=True)
ioArgs.add_argument("-imgo",help="Output well-focused image (offset)",type=str,required=True)
ioArgs.add_argument("-imga",help="Output well-focused image (angle)",type=str,required=True)
# Optional arguments
parser.add_argument("-fx",help="First x sample for window [373]",type=int)
parser.add_argument("-nxw",help="Size of window [512]",type=int)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Create IO for writing at the end
sep = seppy.sep()

# Read in input models
vaxes,vel = sep.read_file(args.vel)
vel = vel.reshape(vaxes.n,order='F')
vel = np.ascontiguousarray(vel).astype('float32')

[nz,nx] = vaxes.n; [dz,dx] = vaxes.d; [oz,ox] = vaxes.o

raxes,ref = sep.read_file(args.ref)
ref = ref.reshape(raxes.n,order='F')
ref = np.ascontiguousarray(ref).astype('float32')

paxes,ptb = sep.read_file(args.ptb)
ptb = ptb.reshape(paxes.n,order='F')
ptb = np.ascontiguousarray(ptb).astype('float32')

# Create migration velocity
velsm = smooth(vel,rect1=30,rect2=30)

# Create velocity with anomaly
velwr = velsm + ptb
plot_imgvelptb(ref,ptb,dz,dx,velmin=-100,velmax=100,thresh=5,agc=False,show=True)

# Acquisition geometry
dsx = 20; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=66,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(velwr,cmap='jet',show=True)

# Create data axes
ntu = 6500; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)

# Model linearized data
dtd = 0.004
allshot = prp.model_lindata(velwr,ref,wav,dtd,verb=True,nthrds=24)

# Taper for migration
prp.build_taper(70,150)

# Wave equation depth migration
img = prp.wem(velwr,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=19)
nh,oh,dh = prp.get_off_axis()

# Window to the region of interest
nxw = args.nxw; nw = args.fx
imgw = img[:,:,nw:nw+nxw]

# Convert to angle gathers
imgang = prp.to_angle(imgw,verb=True,nthrds=24)
na,oa,da = prp.get_ang_axis()

# Window and transpose the image and the label
imgwt = np.transpose(imgw,(0,2,1)) # [nh,nz,nx] -> [nh,nx,nz]

# Write outputs to file
sep.write_file(args.imgo,imgwt.T,ds=[dz,dx,dh],os=[0,0,oh])
sep.write_file(args.imga,imgang.T,ds=[dz,da,dx],os=[0,oa,0])

