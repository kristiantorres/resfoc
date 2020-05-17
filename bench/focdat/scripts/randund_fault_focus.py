"""
Creates a well-focused fault image. Generated image
contains randomly undulating layers and faults

@author: Joseph Jennings
@version: 2020.05.13
"""
import sys, os, argparse, configparser
import numpy as np
import inpout.seppy as seppy
from velocity.stdmodels import undulatingrandfaults2d
from scaas.velocity import create_randomptbs_loc
from utils.rand import randfloat
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
    "mode": 'local',
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
ioArgs.add_argument("-ptb",help="Output velocity perturbation",type=str,required=True)
ioArgs.add_argument("-ref",help="Output reflectivity",type=str,required=True)
ioArgs.add_argument("-lbl",help="Output label",type=str,required=True)
ioArgs.add_argument("-fimgo",help="Focused migrated image (offset)",type=str,required=True)
ioArgs.add_argument("-fimga",help="Focused migrated image (angle)",type=str,required=True)
ioArgs.add_argument("-dimgo",help="Defocused migrated image (offset)",type=str,required=True)
ioArgs.add_argument("-dimga",help="Defocused migrated image (angle)",type=str,required=True)
# Optional arguments
parser.add_argument("-mode",help="Constant or local perturbation (const or [local])",type=str)
parser.add_argument("-fx",help="First x sample for window [373]",type=int)
parser.add_argument("-nxw",help="Size of window [512]",type=int)
parser.add_argument("-na",help="Number of angles to compute [64]",type=int)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Create IO for writing at the end
sep = seppy.sep()

# Create layered model
nx = 1300; nz = 512
vel,ref,cnv,lbl = undulatingrandfaults2d(nz=nz,nx=nx,ofx=0.4,dfx=0.08)
dx = 10; dz = 10

# Create migration velocity
velmig = smooth(vel,rect1=30,rect2=30)

# Create a random perturbation
ano = np.zeros(velmig.shape,dtype='float32')
if(args.mode == 'const'):
  if(np.random.choice([0,1])):
    ano[:,:] = randfloat(0.98,0.995)
  else:
    ano[:,:] = randfloat(1.005,1.02)
elif(args.mode == 'local'):
  ano[:,:] = create_randomptbs_loc(nz,nx,nptbs=3,romin=0.95,romax=1.05,
                                   minnaz=100,maxnaz=150,minnax=100,maxnax=400,mincz=100,maxcz=150,mincx=300,maxcx=900,
                                   mindist=100,nptsz=2,nptsx=2,octaves=2,period=80,persist=0.2,ncpu=1,sigma=20)

# Create velocity with anomaly
veltru = velmig*ano
velptb = veltru - velmig
plot_imgvelptb(ref,-velptb,dz,dx,velmin=-100,velmax=100,thresh=5,agc=False,show=True)

# Acquisition geometry
dsx = 20; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=66,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(veltru,cmap='jet',show=True)

# Create data axes
ntu = 6500; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)

# Model true linearized data
dtd = 0.008
allshot = prp.model_lindata(veltru,ref,wav,dtd,verb=True,nthrds=24)

# Taper for migration
prp.build_taper(70,150)

# Wave equation depth migration (both with anomaly)
imgr = prp.wem(veltru,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=24)
imgw = prp.wem(velmig,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=24)
nh,oh,dh = prp.get_off_axis()

# Window to the region of interest
nxw = args.nxw; nw = args.fx
imgrw = imgr[:,:,nw:nw+nxw]
imgww = imgw[:,:,nw:nw+nxw]
lblw  = lbl[:,:nw:nx+nxw]

# Convert to angle gathers
imgrang = prp.to_angle(imgrw,na=args.na,verb=True,nthrds=24)
imgwang = prp.to_angle(imgww,na=args.na,verb=True,nthrds=24)
na,oa,da = prp.get_ang_axis()

# Window and transpose the image and the label
imgrwt = np.transpose(imgrw,(0,2,1)) # [nh,nz,nx] -> [nh,nx,nz]
imgwwt = np.transpose(imgww,(0,2,1)) # [nh,nz,nx] -> [nh,nx,nz]

# Output inputs
sep.write_file(args.vel,vel,ds=[dz,dx])
sep.write_file(args.ref,ref,ds=[dz,dx])
sep.write_file(args.lbl,lblw,ds=[dz,dx])
sep.write_file(args.ptb,-velptb,ds=[dz,dx])
# Output images
sep.write_file(args.fimgo,imgrwt.T,ds=[dz,dx,dh],os=[0,0,oh])
sep.write_file(args.fimga,imgrang.T,ds=[dz,da,dx],os=[0,oa,0])
sep.write_file(args.dimgo,imgwwt.T,ds=[dz,dx,dh],os=[0,0,oh])
sep.write_file(args.dimga,imgwang.T,ds=[dz,da,dx],os=[0,oa,0])

