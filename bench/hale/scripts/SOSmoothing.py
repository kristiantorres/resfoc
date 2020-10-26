"""
A wrapper for the structure-oriented smoothing Jython script

@author: Joseph Jennings
@version: 2020.10.24
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
import subprocess

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
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
ioArgs.add_argument("-fin",help="input file",type=str)
ioArgs.add_argument("-fout",help="output file",type=str)
# Optional arguments
parser.add_argument("-labels",help="Input fault labels",type=str)
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep()

verb = sep.yn2zoo(args.verb)

if(args.labels is not None):
  laxes,lbl = sep.read_file(args.labels)
  lbl = lbl.reshape(laxes.n,order='F').T
  zidx = lbl == 0
  oidx = lbl == 1
  lbl[zidx] = 1
  lbl[oidx] = -4.6
  smb = smooth(lbl.astype('float32'),rect1=20,rect2=20)
  osmb = "fltsmb.H"
  sep.write_file(osmb,smb.T,ofaxes=laxes)
  sos = "jy ./scripts/sos.py %s %s %s"%(args.fin,osmb,args.fout)
  if(verb): print(sos)
  sp = subprocess.check_call(sos,shell=True)
else:
  sos = "jy ./scripts/sos.py %s %s"%(args.fin,args.fout)
  if(verb): print(sos)
  sp = subprocess.check_call(sos,shell=True)

