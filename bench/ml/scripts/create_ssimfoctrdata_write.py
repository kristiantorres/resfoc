"""
Creates patched data in H5 format from SEPlib files.
The output is a H5 file with images and labels that
is ready to be read in for deep learning

@author: Joseph Jennings
@version: 2020.02.10
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import h5py
import glob
from utils.ptyprint import progressbar, create_inttag
import numpy as np
import deeplearn.utils as dlut
from resfoc.estro import estro_tgt
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "sepdir": None,
    "verb": "y",
    "filelist": "n",
    "prefix": "",
    "norm": 'y',
    "ptchz": 128,
    "ptchx": 128,
    "strdx": 64,
    "strdz": 64,
    "nfiles": 210,
    "blacklist": ['00760'],
    }
if args.conf_file:
  config = configparser.SafeConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))
# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

# Set defaults
parser.set_defaults(**defaults)

# Input files
parser.add_argument("-imgdir",help="Directory containing focused image SEPlib files",type=str,required=True)
parser.add_argument("-resdir",help="Directory containing residual migration SEPlib files",type=str,required=True)
parser.add_argument("-ptbdir",help="Directory containing velocity perturbation SEPlib files",type=str,required=True)
parser.add_argument("-out",help="output h5 file",type=str,required=True)
parser.add_argument("-resprefix",help="Prefix to residual migration files [resfltimg]",type=str,required=True)
parser.add_argument("-ptbprefix",help="Prefix to velocity perturbation files [resfltptb]",type=str,required=True)
parser.add_argument("-imgprefix",help="Prefix to migration files [fltimg]",type=str,required=True)
# Optional arguments
parser.add_argument("-filelist",help="Path to a file list. If one is not provided, one will be created [None]",type=str)
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-ptchz",help="Size of patch in z [128]",type=int)
parser.add_argument("-ptchx",help="Size of patch in x [128]",type=int)
parser.add_argument("-strdz",help="Patch stride (overlap) in z  [64]",type=int)
parser.add_argument("-strdx",help="Patch stride (overlap) in x [64]",type=int)
parser.add_argument("-norm",help="Normalize each patch from -1 to 1 [y]",type=str)
parser.add_argument("-nfiles",help="Number of SEPlib files to use [210]",type=int)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up IO
sep = seppy.sep()

# Get command line arguments
out = args.out
imgprefix = args.imgprefix; 
resprefix = args.resprefix; 
ptbprefix = args.ptbprefix
nfiles = args.nfiles

norm = sep.yn2zoo(args.norm)
verb = sep.yn2zoo(args.verb)

# Get SEPlib files
resfiles = sorted(glob.glob(args.resdir + '/' + resprefix + '-*.H'))
imgfiles = sorted(glob.glob(args.imgdir + '/' + imgprefix + '-*.H'))
ptbfiles = sorted(glob.glob(args.ptbdir + '/' + ptbprefix + '-*.H'))

resmtch = []; ptbmtch = []; imgmtch = []
for iimg in imgfiles:
  # Get the number of the file
  inum = iimg.split(imgprefix+'-')[-1]
  ires = args.resdir + '/' + resprefix + '-' + inum
  iptb = args.ptbdir + '/' + ptbprefix + '-' + inum
  # Append to lists
  if(ires in resfiles and iptb in ptbfiles):
    imgmtch.append(iimg); resmtch.append(ires); ptbmtch.append(iptb)

#resmtch = []; ptbmtch = []; imgmtch = []
#if(args.filelist is None):
#  # Need to find matching files
#  with open("./par/flist.txt",'w') as f:
#    for iimg in imgfiles:
#      # Get the number of the file
#      inum = iimg.split(imgprefix+'-')[-1]
#      ires = args.resdir + '/' + resprefix + '-' + inum
#      iptb = args.ptbdir + '/' + ptbprefix + '-' + inum
#      if(ires in resfiles and iptb in ptbfiles):
#        imgmtch.append(iimg); resmtch.append(ires); ptbmtch.append(iptb)
#        f.write(iimg + ' ' + ires + ' ' + iptb+'\n')
#else:
#  # Read in the file list
#  with open(args.filelist,'r') as f:
#    for line in f.readlines:
#      resmtch.append(line.split(' ')[0])
#      imgmtch.append(line.split(' ')[1])
#      ptbmtch.append(line.split(' ')[2])

