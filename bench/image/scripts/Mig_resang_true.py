"""
Creates a residual migration of a synthetic dataset
with a provided velocity model

@author: Joseph Jennings
@version: 2020.04.04
"""
import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scipy.ndimage import gaussian_filter
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from resfoc.gain import agc,tpow
from resfoc.resmig import preresmig,get_rho_axis
from genutils.plot import plot_wavelet
from genutils.movie import viewimgframeskey
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "y",
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
ioArgs.add_argument("-imgout",help="Output residually-migrated image [nz,nx,nh,nro]",type=str,required=True)
# Required arguments
reqArgs = parser.add_argument_group('Required arguments')
reqArgs.add_argument("-velidx",help="Index of velocity model to use for imaging",type=int,required=True)
# Optional arguments
parser.add_argument("-verb",help="Flag for verbose output ([y] or n)",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Create SEP IO object
sep = seppy.sep()

# Get commandline arguments
idx = args.velidx; verb = sep.yn2zoo(args.verb)

# Read in the model
vaxes,vel = sep.read_file(args.velin)
vel = vel.reshape(vaxes.n,order='F')
velw = np.ascontiguousarray((vel[:,:,idx].T).astype('float32'))
# Read in the reflectivity
raxes,ref = sep.read_file(args.refin)
ref = ref.reshape(raxes.n,order='F')
refw = np.ascontiguousarray((ref[:,:,idx].T).astype('float32'))

# Resample the model
nx = 1024; nz = 512
rvel = (resample(velw,[nx,nz],kind='linear')).T
rref = (resample(refw,[nx,nz],kind='linear')).T
dz = 10; dx = 10

# Create migration velocity
rvelsm = gaussian_filter(rvel,sigma=20)

dsx = 20; bx = 25; bz = 25
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=52,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(rvelsm,cmap='jet',show=False)
prp.plot_acq(rref,cmap='gray',show=False)

# Create data axes
ntu = 6400; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)
plot_wavelet(wav,dtu)

dtd = 0.004
# Model the data
allshot = prp.model_lindata(rvelsm,rref,wav,dtd,verb=verb,nthrds=24)

# Image the data
prp.build_taper(100,200)
prp.plot_taper(rref,cmap='gray')

img = prp.wem(rvelsm,allshot,wav,dtd,nh=16,lap=True,verb=verb,nthrds=24)
nh,oh,dh = prp.get_off_axis()

# Residual migration
inro = 10; idro = 0.0025
storm = preresmig(img,[dh,dz,dx],nps=[2049,513,1025],nro=inro,dro=idro,time=tconv,transp=True,verb=verb,nthreads=19)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

if(aconv):
  # Convert to angle
  stormang = prp.to_angle(storm,oro=ooro,dro=odro,verb=verb,nthrds=24)
  na,oa,da = prp.get_angle_axis()
  # Write to file
  stormangt = np.transpose(stormang,(2,1,3,0))
  if(tconv): dz = dtd
  sep.write_file("resangtrue.H",stormangt,os=[0,oa,0,ooro],ds=[dz,da,dx,odro])
else:
  # Write to file
  stormt = np.transpose(storm,(2,3,1,0))
  if(tconv): dz = dtd
  sep.write_file("resangtrue.H",stormt,os=[0,0,oh,ooro],ds=[dz,dx,dh,odro])

