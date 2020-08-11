"""
Combines separate H5 defocused extended image labels

@author: Joseph Jennings
@version: 2020.05.27
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import glob
import h5py
from utils.ptyprint import progressbar, create_inttag
import numpy as np
from deeplearn.utils import plotseglabel, plotsegprobs, normalize
from deeplearn.dataloader import load_labeled_flat_data
from utils.plot import plot_cubeiso
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "defocanglblprefix": "",
    "defocanglblout": "",
    "nqc": 5,
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
ioArgs.add_argument("-defoclblprefix",help="Defocused Angle H5 prefix",type=str,required=True)
ioArgs.add_argument("-defoclblout",help="Output combined H5 file containing labeled defocused angle patches",type=str,required=True)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
othArgs.add_argument("-nqc",help="Number of focused and defocused patches to QC [5]",type=int)
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
hfot = h5py.File(args.defoclblout,'w')

ntot = 100000

os = [-70.0,0.0,0.0]; ds = [2.22,0.01,0.01]
begex = 0; endex = 0; ctr = 0
for ih5 in h5s:
  print("file=%s"%ih5)
  # Load all data from file
  xblk,yblk = load_labeled_flat_data(ih5,None)
  [nex,nang,nzp,nxp,nchn] = xblk.shape
  endex = begex + nex
  # QC some examples
  for iqc in progressbar(range(args.nqc), "nqc:"):
    idx = np.random.randint(nex)
    plot_cubeiso(xblk[idx,:,:,:,0],os=os,ds=ds,stack=True,elev=15,show=True,verb=False,title='%d'%idx)
  # Write it
  k = 0
  if(verb): print(begex,endex)
  for iex in progressbar(range(begex,endex), "nex:"):
    datatag = create_inttag(iex,ntot)
    hfot.create_dataset("x"+datatag, (nang,nzp,nxp,1), data=np.expand_dims(xblk[k],axis=-1), dtype=np.float32)
    hfot.create_dataset("y"+datatag, (1,), data=yblk[k], dtype=np.float32)
    k += 1; ctr += 1
  begex = ctr

hfot.close()
