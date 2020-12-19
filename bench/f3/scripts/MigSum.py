"""
Sums over extended images written to disk

@author: Joseph Jennings
@version: 2020.12.15
"""
import sys, os, glob, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from genutils.ptyprint import progressbar

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "y",
    "logfile": None
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
ioArgs.add_argument("-migdir",help="input directory",type=str)
ioArgs.add_argument("-fout",help="output file",type=str)
# Optional arguments
parser.add_argument("-logfile",help="Output logfile for progressbar",type=str)
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep()
verb = sep.yn2zoo(args.verb)
if(verb):
  logfile = args.logfile
  if(logfile is None):
    logfile = args.migdir[:-1] + '-log.txt'
    f = open(logfile,'w')

# Get all images in the directory
if(not os.path.exists(args.migdir)):
  print("Directory %s does not exist"%(migdir))
  sys.exit()
pmigs = glob.glob(args.migdir + "/*.H")
if(len(pmigs) == 0):
  print("No files are present within %s. Exiting..."%(args.migdir))
  sys.exit()

# Get the axes of the output
paxes = sep.read_header(pmigs[0])
nx,ny,nz,nhx = paxes.n
ox,oy,oz,ohx = paxes.o
dx,dy,dz,dhx = paxes.d

# Read in the output file if it already exists
if(os.path.exists(args.fout)):
  iaxes,img = sep.read_file(args.fout)
  img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32') # [nhx,ny,nx,nz]
else:
  img = np.zeros([nhx,ny,nx,nz],dtype='float32')

# Loop over each image and add it to img
for iimg in progressbar(range(len(pmigs)),"eimg:",file=f,verb=verb):
  # Read in the file
  paxes,prt = sep.read_file(pmigs[iimg])
  prt  = np.ascontiguousarray(prt.reshape(paxes.n,order='F').T).astype('float32') # [nhx,nz,ny,nx]
  img += np.ascontiguousarray(np.transpose(prt,(0,2,3,1))) # [nhx,nz,ny,nx] -> [nhx,ny,nx,nz]

# Close the log file
f.close()

# Write out the reduced array
sep.write_file(args.fout,img.T,os=[oz,ox,oy,ohx],ds=[dz,dx,dy,dhx])

