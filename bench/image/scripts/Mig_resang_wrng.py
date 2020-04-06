"""
Creates a prestack residually depth migrated image
for when migrated with the wrong velocity model
@author: Joseph Jennings
@version: 2020.04.04
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scipy.ndimage import gaussian_filter
import scaas.defaultgeom as geom
from scaas.velocity import create_randomptb_loc
from scaas.wavelet import ricker
from resfoc.gain import agc,tpow
from resfoc.resmig import preresmig,get_rho_axis
from utils.plot import plot_wavelet
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "y",
    "nthreads": 24,
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
ioArgs.add_argument("-velin",help="Input velocity model",type=str,required=True)
ioArgs.add_argument("-refin",help="Input reflectivity model",type=str,required=True)
ioArgs.add_argument("-ptbout",help="Output velocity perturbation",type=str,required=True)
ioArgs.add_argument("-imgout",help="Output migrated image [nz,nx,nh,nro]",type=str,required=True)
# Required arguments
reqArgs = parser.add_argument_group('Required arguments')
reqArgs.add_argument("-velidx",help="Index of velocity model to use for imaging",type=int,required=True)
# Optional arguments
parser.add_argument("-verb",help="Flag for verbose output ([y] or n)",type=str)
parser.add_argument("-nthreads",help="Number of CPU threads [24]",type=int)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Create SEP IO object
sep = seppy.sep()

# Get command line arguments
verb = sep.yn2zoo(args.verb); nthrd = args.nthreads

# Read in the model
vaxes,vel = sep.read_file(args.velin)
vel = vel.reshape(vaxes.n,order='F')
velw = np.ascontiguousarray((vel[:,:,0].T).astype('float32'))
# Read in the reflectivity
raxes,ref = sep.read_file(args.refin)
ref = ref.reshape(raxes.n,order='F')
refw = np.ascontiguousarray((ref[:,:,0].T).astype('float32'))

# Resample the model
nx = 1024; nz = 512
rvel = (resample(velw,[nx,nz],kind='linear')).T
rref = (resample(refw,[nx,nz],kind='linear')).T
dz = 10; dx = 10

# Create migration velocity
rvelsm = gaussian_filter(rvel,sigma=20)

# Scale by a random perturbation
nro1=3; oro1=1.03; dro1=0.01
romin1 = oro1 - (nro1-1)*dro1; romax1 = romin1 + dro1*(2*nro1-1)
#TODO: choose randomly the position of the anomaly (within bounds)
# Make sure not too close together
rhosm1 = create_randomptb_loc(nz,nx,romin1,romax1,naz=150,nax=150,cz=200,cx=700,
    nptsz=2,nptsx=2,octaves=3,period=80,Ngrad=80,persist=0.2,ncpu=1)

nro2=3; oro2=0.97; dro2=0.01
romin2 = oro2 - (nro2-1)*dro1; romax2 = romin2 + dro2*(2*nro2-1)
rhosm2 = create_randomptb_loc(nz,nx,romin2,romax2,naz=150,nax=150,cz=120,cx=300,
    nptsz=2,nptsx=2,octaves=3,period=80,Ngrad=80,persist=0.2,ncpu=1)
# Plot for QC
plt.figure(1); plt.imshow(rvelsm,cmap='jet')
plt.figure(2); plt.imshow(rvelsm*rhosm1*rhosm2,cmap='jet')
plt.figure(3); plt.imshow(rhosm1*rhosm2,cmap='jet')
plt.show()
rvelwr = rvelsm*rhosm1*rhosm2

dsx = 20; bx = 25; bz = 25
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=52,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(rvelwr,cmap='jet',show=False)
prp.plot_acq(rref,cmap='gray',show=False)

# Create data axes
ntu = 6400; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)
plot_wavelet(wav,dtu)

dtd = 0.004
# Model the data
allshot = prp.model_lindata(rvelsm,rref,wav,dtd,verb=verb,nthrds=nthrd)

# Image the data
prp.build_taper(100,200)
prp.plot_taper(rref,cmap='gray')

img = prp.wem(rvelwr,allshot,wav,dtd,nh=16,lap=True,verb=verb,nthrds=nthrd)
nh,oh,dh = prp.get_off_axis()

# Residual migration
inro = 10; idro = 0.0025
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)
if(nthrd > onro): nthrd = onro
storm = preresmig(img,[dh,dz,dx],nps=[2049,513,1025],nro=inro,dro=idro,time=False,transp=True,verb=verb,nthreads=nthrd)

# Write image to file
stormt = np.transpose(storm,(2,3,1,0))
sep.write_file(args.imgout,stormt,os=[0,0,oh,ooro],ds=[dz,dx,dh,odro])
# Write velocity perturbation to file
sep.write_file(args.ptbout,rvelsm-rvelwr,ds=[dz,dx])

