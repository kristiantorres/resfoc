"""
Cats up to 10,000 .H (SEPlib) files at at time
using the SEP program Cat3d

@author: Joseph Jennings
@version: 2019.12.31
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import subprocess, glob

def chunks(l, n):
  """ Yield sucessive n-sized chunks from l"""
  for i in range(0, len(l), n):
    yield l[i:i + n]

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    "klean": "y",
    "axis": 3,
    "dir": " ",
    "files": [],
    }
if args.conf_file:
  config = ConfigParser.SafeConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("Defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.set_defaults(**defaults)
# Input files
parser.add_argument("out=",help="Catted file")
# Required arguments
requiredNamed = parser.add_argument_group('required parameters')
# Optional arguments
parser.add_argument("-files",help="Files to be catted as 'file1.H file2.H ...'",type=str)
parser.add_argument("-dir",help="Directory containing files",type=str)
parser.add_argument("-axis",help="Axis along which to cat [3]",type=int)
parser.add_argument("-klean",help="Clean files ([y] or n)",type=str)
parser.add_argument("-verb",help="Verbosity flag (y or n)",type=str)
args = parser.parse_args(remaining_argv)

# Setup IO
sep = seppy.sep(sys.argv)

# Get command line arguments
odir = args.dir
assert(len(args.files) != 0 or odir != " "), "Must provide either file list or directory"
if(len(args.files) != 0):
  files = args.files.split(' ')
else:
  files = sorted(glob.glob(odir + "/*.H"))

axis = args.axis
verb = sep.yn2zoo(args.verb)
klean = sep.yn2zoo(args.klean)

# Set up SEP
ofname  = sep.get_fname("out")
bname = os.path.splitext(ofname)[0]

csize = 10
if(len(files) < csize):
  # Run normal cat
  cat = "Cat axis=%d "%(axis)
  cat += " ".join(files) + " > %s"%(ofname)
  if(verb): print(cat)
  sp = subprocess.check_call(cat,shell=True)
else:
  # Split files into bunches
  bunches = list(chunks(files,csize))
  ib = 0
  ocats = []
  # Cat each bunch
  for ibnch in bunches:
    cat = "Cat axis=%d "%(axis)
    cat += " ".join(ibnch) + " > %s-cat%d.H"%(bname,ib)
    if(verb): print(cat)
    sp = subprocess.check_call(cat,shell=True)
    ocats.append("%s-cat%d.H"%(bname,ib))
    ib += 1
  # Cat the catted files
  #cat = "Cat virtual=1 axis=%d "%(axis)
  #cat += " ".join(ocats) + " > %s"%(ofname)
  #if(verb): print(cat)
  #sp = subprocess.check_call(cat,shell=True)

# Clean up the intermediate catted files
if(klean):
  rm = 'Rm *-cat*.H'
  if(verb): print(rm)
  sp = subprocess.check_call(rm,shell=True)

