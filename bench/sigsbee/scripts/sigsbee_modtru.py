"""
Migrates the sigsbee model with the correct velocity

@author: Joseph Jennings
@version: 2020.05.13
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
    "na": 64,
    "mode": 'local',
    "nxo": -1,
    "nzo": -1,
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
ioArgs.add_argument("-dat",help="Output data",type=str,required=True)
# Optional arguments
parser.add_argument("-nxo",help="Output lateral size of migration velocity and reflectivity",type=int)
parser.add_argument("-nzo",help="Output depth size of migration velocity and reflectivity",type=int)
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

# First read in sigsbee model
vaxes,vel = sep.read_file("./dat/vmig.H",form='native')
vel = vel.reshape(vaxes.n,order='F')
raxes,ref = sep.read_file("./dat/ref.H",form='native')
ref = ref.reshape(raxes.n,order='F')

# Get the new output shapes

# Window the first sample of reflectivity
refw = ref[:,1:]

nzr,nxr = refw.shape; dzr,dxr = raxes.d

# First resample the migration velocity so it is same size
velr = resample(vel,[nzr,nxr],kind='cubic')

# Now resample them to desired output
nxo = args.nxo; nzo = args.nzo
if(nxo == -1): nxo = nxr
if(nxo == -1): nzo = nzr

velo,[dzo,dxo] = resample(velr,[nzo,nxo],ds=[dzr,dxr],kind='cubic')
refo,[dzo,dxo] = resample(refw,[nzo,nxo],ds=[dzr,dxr],kind='cubic')

# Window and get samplings
vel = velo[200:,:]; ref = refo[200:,:]
print(vel.shape)
nz,nx = vel.shape
#dz = dzo; dx = dxo
dz = 0.01; dx = 0.01

# Acquisition geometry
dsx = 20; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=161,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(vel,cmap='jet',show=False)
prp.plot_acq(ref,cmap='gray',show=True,vmin=-0.1,vmax=0.1)

# Create data axes
ntu = 10000; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)

# Model true linearized data
dtd = 0.008
allshot = prp.model_lindata(vel,ref,wav,dtd,verb=True,nthrds=24)

# Output inputs
sep.write_file(args.vel,vel,ds=[dz,dx])
sep.write_file(args.ref,ref,ds=[dz,dx])
# Output data
allshott = np.transpose(allshot,(0,2,1)) # [nsx,nt,nrx] -> [nsx,nrx,nt]
sep.write_file(args.dat,allshott.T,ds=[dtd,dx,dsx])

