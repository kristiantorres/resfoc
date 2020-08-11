"""
Combines separate H5 defocused image labels

@author: Joseph Jennings
@version: 2020.05.17
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import glob
import h5py
from utils.ptyprint import progressbar, create_inttag
import numpy as np
from deeplearn.utils import plotseglabel, plotsegprobs, normalize
from deeplearn.dataloader import load_allssimcleandata
from deeplearn.focuslabels import corrsim
import random
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "defoclblprefix": "",
    "defoclblout": "",
    }
if args.conf_file:
  config = configparser.ConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.set_defaults(**defaults)
# IO
ioArgs = parser.add_argument_group('Input/Output')
ioArgs.add_argument("-defoclblprefix",help="Defocused H5 prefix",type=str,required=True)
ioArgs.add_argument("-defoclblout",help="Output combined H5 file containing labeled defocused patches",type=str,required=True)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
othArgs.add_argument("-nqc",help="Number of focused and defocused patches to QC",type=int)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Setup IO
sep = seppy.sep()

# Flags
verb =  sep.yn2zoo(args.verb)

h5s = sorted(glob.glob(args.defoclblprefix + '*.h5'))

allx = []; ally = []
for ih5 in h5s:
  print("file=%s"%ih5)
  xblk,yblk = load_allssimcleandata(ih5,None)
  allx.append(xblk); ally.append(yblk)

# Convert to numpy array and write out
allx = np.concatenate(allx,axis=0)
ally = np.concatenate(ally,axis=0)

print(allx.shape)
print(ally.shape)
ntot = allx.shape[0]; nzp = allx.shape[1]; nxp = allx.shape[2]

hfot = h5py.File(args.defoclblout,'w')
# Write to one large dataset
for iex in progressbar(range(ntot), "nex:"):
  datatag = create_inttag(iex,ntot)
  hfot.create_dataset("x"+datatag, (nxp,nzp,1), data=np.expand_dims(allx[iex],axis=-1), dtype=np.float32)
  hfot.create_dataset("y"+datatag, (1,), data=ally[iex], dtype=np.float32)

hfot.close()
