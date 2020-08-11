"""
Creates simple layered with three faults training data

Outputs the following:
  (a) A layered v(z) velocity model with three faults
  (b) The associated reflecitivity
  (c) The associated fault labels
  (d) A velocity anomaly
  (e) Prestack residual migration (subsurface offsets)
  (f) Angle stack residual migration
  (g) Semblance computed from angle gathers

@author: Joseph Jennings
@version: 2020.05.02
"""
import sys, os, argparse, configparser
import numpy as np
import inpout.seppy as seppy
from velocity.stdmodels import layeredfaults2d
from scaas.velocity import create_randomptbs_loc
from scaas.trismooth import smooth
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from resfoc.resmig import preresmig,convert2time,get_rho_axis,convert2time
from resfoc.gain import agc
from joblib import Parallel, delayed

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
ioArgs.add_argument("-vel",help="Output velocity model",type=str,required=True)
ioArgs.add_argument("-ref",help="Output reflectivity",type=str,required=True)
ioArgs.add_argument("-ptb",help="Output velocity anomaly",type=str,required=True)
ioArgs.add_argument("-lbl",help="Output windowed label",type=str,required=True)
ioArgs.add_argument("-res",help="Output residual migration (offset)",type=str,required=True)
ioArgs.add_argument("-stk",help="Output residual migration (angle stack)",type=str,required=True)
ioArgs.add_argument("-smb",help="Output semblance",type=str,required=True)
ioArgs.add_argument("-dpath",help="Output datapath",type=str,required=True)
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

# Create layered model
nx = 1300; nz = 512
vel,ref,cnv,lbl = layeredfaults2d(nz=nz,nx=nx,ofx=0.4,dfx=0.08)
dx = 10; dz = 10

# Create migration velocity
velsm = smooth(vel,rect1=30,rect2=30)

# Create a random perturbation
ano = create_randomptbs_loc(nz,nx,nptbs=3,romin=0.95,romax=1.00,
                            minnaz=100,maxnaz=150,minnax=100,maxnax=400,mincz=100,maxcz=150,mincx=300,maxcx=800,
                            mindist=100,nptsz=2,nptsx=2,octaves=2,period=80,persist=0.2,ncpu=1,sigma=20)

# Create velocity with anomaly
velwr = velsm*ano
velptb = velwr - velsm

# Acquisition geometry
dsx = 20; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=66,dsx=dsx,bx=bx,bz=bz)

# Create data axes
ntu = 6500; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)

# Model linearized data
dtd = 0.004
allshot = prp.model_lindata(velsm,ref,wav,dtd,verb=True,nthrds=24)

# Taper for migration
prp.build_taper(70,150)

# Wave equation depth migration
img = prp.wem(velwr,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=12)
nh,oh,dh = prp.get_off_axis()

# Window and transpose the image and the label
nxw = args.nxw; nw = args.fx
imgw = img[:,:,nw:nw+nxw]
imgwt = np.transpose(imgw,(0,2,1)) # [nh,nz,nx] -> [nh,nx,nz]
lblw = lbl[:,nw:nw+nxw]

# Depth Residual migration
inro = 17; idro = 0.00125
rmig = preresmig(imgwt,[dh,dx,dz],nps=[2049,1025,513],nro=inro,dro=idro,time=False,nthreads=10,verb=True)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

# Conversion to time
rmigt = convert2time(rmig,dz,dt=dtd,dro=odro,verb=True)

# Convert to angle
rmigtt = np.ascontiguousarray(np.transpose(rmigt,(0,1,3,2))).astype('float32') # [nro,nh,nx,nz] -> [nro,nx,nh,nz]
stormang = prp.to_angle(rmigtt,oro=ooro,dro=odro,verb=True,nthrds=24)
na,oa,da = prp.get_ang_axis()

# Compute the stack
stormangstk = np.sum(stormang,axis=2)

# Gain the data before semblance
angagc = np.asarray(Parallel(n_jobs=10)(delayed(agc)(stormang[iro]) for iro in range(onro)))

# Compute semblance
stackg   = np.sum(angagc,axis=2)
stacksq = stackg*stackg
num = smooth(stacksq.astype('float32'),rect1=10,rect3=3)

sqstack = np.sum(angagc*angagc,axis=2)
denom = smooth(sqstack.astype('float32'),rect1=10,rect3=3)

semb = num/denom
sembt = np.transpose(semb,(2,0,1)) # [nro,nx,nz] -> [nz,nro,nx]

# Write outputs to file
sep.write_file(args.vel,vel,ds=[dz,dx],dpath=args.dpath+'/')
sep.write_file(args.ref,ref,ds=[dz,dx],dpath=args.dpath+'/')
sep.write_file(args.lbl,lblw,ds=[dz,dx],dpath=args.dpath+'/')
sep.write_file(args.ptb,velptb,ds=[dz,dx],dpath=args.dpath+'/')
sep.write_file(args.res,rmig.T,ds=[dz,dx,dh,odro],os=[0,0,oh,ooro],dpath=args.dpath+'/')
sep.write_file(args.stk,stormangstk.T,ds=[dz,dx,odro],os=[0,0,ooro],dpath=args.dpath+'/')
sep.write_file(args.smb,sembt,ds=[dz,odro,dx],os=[0,ooro,0],dpath=args.dpath+'/')

# Success flag for cluster manager
print("Success!")
