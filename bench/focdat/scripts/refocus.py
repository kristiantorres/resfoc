"""
Reads in a defocused image residually migrates it
and refocuses it with semblance

@author: Joseph Jennings
@version: 2020.05.13
"""
import sys, os, argparse, configparser
import numpy as np
import inpout.seppy as seppy
from scaas.trismooth import smooth
from scaas.off2ang import off2ang,get_ang_axis
from resfoc.resmig import preresmig,convert2time,get_rho_axis
from resfoc.gain import agc
from joblib import Parallel, delayed
import subprocess
from resfoc.estro import refocusimg

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
ioArgs.add_argument("-stormang",help="Input residually migrated angle gathers",type=str,required=True)
ioArgs.add_argument("-stk",help="Output residual migration (angle stack)",type=str,required=True)
ioArgs.add_argument("-smb",help="Output semblance",type=str,required=True)
ioArgs.add_argument("-rho",help="Output semblance picks (z,rho)",type=str,required=True)
ioArgs.add_argument("-rfi",help="Output refocused image",type=str,required=True)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Create IO for writing at the end
sep = seppy.sep()

# Read in residually-migrated angle gathers
saxes,stormang = sep.read_file(args.stormang)
stormang = np.ascontiguousarray(stormang.reshape(saxes.n,order='F').T).astype('float32')

# Get dimensions
[nz,na,nx,nro] = saxes.n; [dz,da,dx,dro] = saxes.d; [oz,oa,ox,oro] = saxes.o

# Compute the stack
stormangstk = np.sum(stormang,axis=2)

# Gain the data before semblance
angagc = np.asarray(Parallel(n_jobs=24)(delayed(agc)(stormang[iro]) for iro in range(nro)))

# Compute semblance
stackg   = np.sum(angagc,axis=2)
stacksq = stackg*stackg
num = smooth(stacksq.astype('float32'),rect1=10,rect3=3)

sqstack = np.sum(angagc*angagc,axis=2)
denom = smooth(sqstack.astype('float32'),rect1=10,rect3=3)

semb = num/denom
sembt = np.transpose(semb,(2,0,1)) # [nro,nx,nz] -> [nz,nro,nx]

# Write outputs to file
sep.write_file(args.stk,stormangstk.T,ds=[dz,dx,dro],os=[0,0,oro])
sep.write_file(args.smb,sembt,ds=[dz,dro,dx],os=[0,oro,0])

# Normalize and pick semblance
sp = subprocess.check_call("Scale scale_to=1 < %s > smbnorm.H"%(args.smb),shell=True)
sp = subprocess.check_call("sfpick vel0=1.0 rect1=40 rect2=20 < smbnorm.H > %s"%(args.rho),shell=True)
sp = subprocess.check_call("Rm smbnorm.H",shell=True)

# Read in picks and perform refocusing
raxes,rimg = sep.read_file(args.rho)
rimg = rimg.reshape(raxes.n,order='F')
rimg = np.ascontiguousarray(rimg.T).astype('float32')

# Refocus the image with the picks
rfis = refocusimg(stormangstk,rimg,dro)
sep.write_file(args.rfi,rfis.T,ds=[dz,dx])

