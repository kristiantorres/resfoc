"""
Defocuses a focused image via residual migration

@author: Joseph Jennings
@version: 2020.05.16
"""
import sys, os, argparse, configparser
import numpy as np
import inpout.seppy as seppy
from resfoc.resmig import preresmig,get_rho_axis,convert2time
from scaas.off2ang import off2ang,get_ang_axis

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": 'y',
    "na": 64,
    "oro": 1.0,
    "dro": 0.00125,
    "nro": 21,
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
ioArgs.add_argument("-img",help="Input prestack focused image",type=str,required=True)
ioArgs.add_argument("-reso",help="Output residually migrated offset image",type=str,required=True)
ioArgs.add_argument("-resa",help="Output residually migrated angle image",type=str,required=True)
ioArgs.add_argument("-dpath",help="Datapath for writing binary files",type=str,required=True)
# Optional arguments
parser.add_argument("-verb",help="Size of window [512]",type=str)
parser.add_argument("-na",help="Number of angles to compute [64]",type=int)
parser.add_argument("-nro",help="Number of residual migrations from which to choose",type=int)
parser.add_argument("-oro",help="Center residual migration value [1.0]",type=float)
parser.add_argument("-dro",help="Residual migration sampling",type=float)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Create IO object
sep = seppy.sep()

# Read in the defocused image
iaxes,img = sep.read_file(args.img)
img  = img.reshape(iaxes.n,order='F')
imgt = np.ascontiguousarray(img.T).astype('float32')

# Get axes
[nz,nx,nh] = iaxes.n; [oz,ox,oh] = iaxes.o; [dz,dx,dh] = iaxes.d

# Residual migration parameters
oro = args.oro; nro = args.nro; dro = args.dro

# Compute all of the rhos
foro = oro - (nro-1)*dro; fnro = 2*nro-1
rhos = np.linspace(foro,foro + (fnro-1)*dro,2*nro-1)
# Choose a rho for defocusing
if(np.random.choice([0,1])):
  rho = np.random.randint(0,nro)*dro + foro
else:
  rho = np.random.randint(nro+1,fnro)*dro + foro

# Depth Residual migration for one rho
rmig = preresmig(imgt,[dh,dx,dz],nps=[2049,1025,513],nro=1,oro=rho,dro=dro,time=False,nthreads=1,verb=True)
onro,ooro,odro = get_rho_axis(nro=nro,dro=dro)

# Conversion to time
dtd = 0.004
rmigt = convert2time(rmig,dz,dt=dtd,oro=rho,dro=odro,verb=True)

# Convert to angle
rmigtt = np.ascontiguousarray(np.transpose(rmigt,(0,1,3,2))).astype('float32') # [nro,nh,nx,nz] -> [nro,nh,nz,nx]
stormang = off2ang(rmigtt,oh,dh,dz,na=args.na,oro=ooro,dro=odro,verb=True,nthrds=24)
na,oa,da = get_ang_axis(na=args.na)

# Write to file
sep.write_file(args.reso,rmig.T,ds=[dz,dx,dh],os=[0,0,oh],dpath=args.dpath+'/')
sep.write_file(args.resa,stormang.T,ds=[dz,da,dx],os=[0,oa,0],dpath=args.dpath+'/')

# Write the residual migration parameter to the files
with open(args.reso,'a') as f:
  f.write('rho=%f\n'%(rho))

with open(args.resa,'a') as f:
  f.write('rho=%f\n'%(rho))

# Flag for cluster manager
print("Success!")

