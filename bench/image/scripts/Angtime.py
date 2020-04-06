"""
Converts prestack-depth residually migrated images
and converts them to angle or time
@author: Joseph Jennings
@version: 2020.04.04
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from scaas.off2ang import off2ang,get_ang_axis
from resfoc.resmig import convert2time

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "toangle": "y",
    "totime": "n",
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
ioArgs.add_argument("-fin",help="Input prestack residual depth migration [nz,nx,nh,nro]",type=str,required=True)
ioArgs.add_argument("-out",help="Image converted to angle/time or both",type=str,required=True)
# Optional arguments
parser.add_argument("-toangle",help="Flag to convert to angle ([y] or n)",type=str)
parser.add_argument("-totime",help="Flag to convert to time (y or [n])",type=str)
parser.add_argument("-nthreads",help="Number of CPU threads [24]",type=int)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# SEP IO object
sep = seppy.sep()
fin = args.fin; fout = args.out

# Get arguments
tconv = sep.yn2zoo(args.totime); aconv = sep.yn2zoo(args.toangle)
nthrd = args.nthreads

# Read in the subsurface offset image
iaxes,imgz = sep.read_file(fin)
imgz = imgz.reshape(iaxes.n,order='F')
[dz,dx,dh,dro] = iaxes.d; [oz,ox,oh,oro] = iaxes.o

if(tconv):
  imgzt = np.ascontiguousarray(np.transpose(imgz,(3,2,1,0))).astype('float32')
  # Convert to time
  dt = 0.004
  imgt = convert2time(imgzt,dz,dt=dt,dro=dro,verb=True)
  if(aconv):
    # Convert to angle
    imgtt = np.ascontiguousarray(np.transpose(imgt,(0,1,3,2))).astype('float32')
    stormang = off2ang(imgtt,oh,dh,dz,oro=oro,dro=dro,transp=True,verb=True,nthrds=nthrd)
    na,oa,da = get_ang_axis()
    # Write to file
    stormangt = np.transpose(stormang,(2,1,3,0))
    sep.write_file(fout,stormangt,os=[oz,oa,ox,oro],ds=[dt,da,dx,dro])
  else:
    # Write to file
    imgtt = np.ascontiguousarray(np.transpose(imgt,(3,2,1,0))).astype('float32')
    sep.write_file(fout,imgtt,os=[oz,ox,oh,oro],ds=[dt,dx,dh,dro])
else:
  if(aconv):
    # Convert to angle
    imgzt = np.ascontiguousarray(np.transpose(imgz,(3,2,0,1))).astype('float32')
    stormang = off2ang(imgzt,oh,dh,dz,oro=oro,dro=dro,transp=True,verb=True,nthrds=nthrd)
    na,oa,da = get_ang_axis()
    # Write to file
    stormangt = np.transpose(stormang,(2,1,3,0))
    sep.write_file(fout,stormangt,os=[oz,oa,ox,oro],ds=[dz,da,dx,dro])
  else:
    # Write to file
    sep.write_file(fout,imgz,os=[oz,ox,oh,oro],ds=[dz,dx,dh,dro])

