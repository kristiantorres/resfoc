"""
Splits an H5 file into two other H5 files.
Useful for splitting training data into training
and validation sets

The split amount is determined by the user and
and can either be done randomly or take the first
X% as file 1 and the remaining as file2

Note that this also will remove the input file

@author: Joseph Jennings
@version: 2020.01.04
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from deeplearn.dataloader import splith5

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    "split": 0.8,
    "random": "n",
    "klean": "y",
    }
if args.conf_file:
  config = ConfigParser.SafeConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("Defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

# Set defaults
parser.set_defaults(**defaults)

# Input files
ioArgs = parser.add_argument_group('Inputs and outputs')
ioArgs.add_argument("-hfin",help="Input H5 file",type=str)
ioArgs.add_argument("-hf1",help="Output file 1",type=str)
ioArgs.add_argument("-hf2",help="Output file 2 ",type=str)
# Optional arguments
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
parser.add_argument("-klean",help="Remove the input file ([y] or n)",type=str)
parser.add_argument("-split",help="Split percentage [0.8]",type=float)
parser.add_argument("-rand",help="Randomly split the file [n]",type=str)
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
clean = sep.yn2zoo(args.klean)
rand  = sep.yn2zoo(args.random)
split = args.split

# Call the splitting function
splith5(args.hfin,args.hf1,args.hf2,split,rand,clean)

