"""
Creates a prestack residually depth migrated image
for when migrated with the wrong velocity model
@author: Joseph Jennings
@version: 2020.04.09
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
from scipy.ndimage import gaussian_filter
import scaas.defaultgeom as geom
from scaas.velocity import create_randomptb_loc, create_randomptbs_loc
from scaas.wavelet import ricker
from resfoc.gain import agc,tpow
from resfoc.resmig import preresmig,get_rho_axis
from genutils.ptyprint import create_inttag

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "y",
    "nthreads": 16,
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
ioArgs.add_argument("-outdir",help="Output directory of where to write headers",type=str,required=True)
ioArgs.add_argument("-dpath",help="Datapath of where to write binaries",type=str,required=True)
ioArgs.add_argument("-ptbpf",help="Prefix of output velocity perturbation",type=str,required=True)
ioArgs.add_argument("-imgpf",help="Prefix of output migrated image [nz,nx,nh,nro]",type=str,required=True)
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
velw = np.ascontiguousarray((vel[:,:,args.velidx].T).astype('float32'))
# Read in the reflectivity
raxes,ref = sep.read_file(args.refin)
ref = ref.reshape(raxes.n,order='F')
refw = np.ascontiguousarray((ref[:,:,args.velidx].T).astype('float32'))

# Resample the model
nx = 1024; nz = 512
rvel = (resample(velw,[nx,nz],kind='linear')).T
rref = (resample(refw,[nx,nz],kind='linear')).T
dz = 10; dx = 10

# Create migration velocity
rvelsm = gaussian_filter(rvel,sigma=20)

# Scale by a random perturbation
rho = create_randomptbs_loc(nz,nx,nptbs=3,romin=0.95,romax=1.06,
                            minnaz=50,maxnaz=300,minnax=50,maxnax=300,mincz=150,maxcz=300,mincx=150,maxcx=850,
                            mindist=100,nptsz=2,nptsx=2,octaves=2,period=80,Ngrad=80,persist=0.2,ncpu=1,sigma=20)

rvelwr = rvelsm*rho

dsx = 20; bx = 25; bz = 25
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=52,dsx=dsx,bx=bx,bz=bz)

# Create data axes
ntu = 6400; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)

dtd = 0.004
# Model the data
allshot = prp.model_lindata(rvelsm,rref,wav,dtd,verb=verb,nthrds=nthrd)

# Image the data
prp.build_taper(100,200)

img = prp.wem(rvelwr,allshot,wav,dtd,nh=16,lap=True,verb=verb,nthrds=12)
nh,oh,dh = prp.get_off_axis()

# Residual migration
inro = 10; idro = 0.0025
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)
if(nthrd > onro): nthrd = onro
storm = preresmig(img,[dh,dz,dx],nps=[2049,513,1025],nro=inro,dro=idro,time=False,transp=True,verb=verb,nthreads=10)

# Prepare output names
tot = args.velin.split('/')[-1].split('.')[0]
base = tot.rstrip('0123456789')
num = create_inttag(int(tot[len(base):])+args.velidx,10000)
iname = args.outdir + '/' + args.imgpf + '-%s.H'%(num)
pname = args.outdir + '/' + args.ptbpf + '-%s.H'%(num)

# Write image to file
stormt = np.transpose(storm,(2,3,1,0))
sep.write_file(iname,stormt,os=[0,0,oh,ooro],ds=[dz,dx,dh,odro],dpath=args.dpath+'/')
# Write velocity perturbation to file
sep.write_file(pname,rvelsm-rvelwr,ds=[dz,dx],dpath=args.dpath+'/')

# Flag for cluster manager to determine success
print("Success!")
