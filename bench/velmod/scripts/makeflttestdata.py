"""
Creates test data for testing segmentation
of migrated and residually migrated images

@author: Joseph Jennings
@version: 2020.04.14
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import h5py
import glob
from utils.ptyprint import progressbar, create_inttag
import numpy as np
import deeplearn.utils as dlut
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "y",
    "filelist": "n",
    "prefix": "",
    "nfiles": 210,
    "filebeg": 0,
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
parser.add_argument("-veldir",help="Directory containing velocity model SEPlib files",type=str,required=True)
parser.add_argument("-imgdir",help="Directory containing focused image SEPlib files",type=str,required=True)
parser.add_argument("-resdir",help="Directory containing residual migration SEPlib files",type=str,required=True)
parser.add_argument("-ptbdir",help="Directory containing velocity perturbation SEPlib files",type=str,required=True)
parser.add_argument("-outlbl",help="Output velocity/fault h5 file",type=str,required=True)
parser.add_argument("-outimg",help="Output image h5 file",type=str,required=True)
parser.add_argument("-outres",help="Output residual migration h5 file",type=str,required=True)
parser.add_argument("-outptb",help="Output residual migration h5 file",type=str,required=True)
parser.add_argument("-outcnv",help="Output residual migration h5 file",type=str,required=True)
parser.add_argument("-resprefix",help="Prefix to residual migration files [resfltimg]",type=str,required=True)
parser.add_argument("-ptbprefix",help="Prefix to velocity perturbation files [resfltptb]",type=str,required=True)
parser.add_argument("-imgprefix",help="Prefix to migration files [fltimg]",type=str,required=True)
parser.add_argument("-filebeg",help="First file from which to start taking data [0]",type=int)
# Optional arguments
parser.add_argument("-filelist",help="Path to a file list. If one is not provided, one will be created [None]",type=str)
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-nfiles",help="Number of SEPlib files to use [210]",type=int)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up IO
sep = seppy.sep()

# Get command line arguments
outlbl = args.outlbl
outimg = args.outimg
outres = args.outres
outptb = args.outptb
outcnv = args.outcnv
imgprefix = args.imgprefix
resprefix = args.resprefix
ptbprefix = args.ptbprefix
nfiles = args.nfiles
filebeg = args.filebeg

verb = sep.yn2zoo(args.verb)

# Get SEPlib files
resfiles = sorted(glob.glob(args.resdir + '/' + resprefix + '-*.H'))
imgfiles = sorted(glob.glob(args.imgdir + '/' + imgprefix + '-*.H'))
ptbfiles = sorted(glob.glob(args.ptbdir + '/' + ptbprefix + '-*.H'))
fltfiles = sorted(glob.glob(args.veldir + '/' + 'velfltlbl*.H'))[filebeg:filebeg+nfiles]
cnvfiles = sorted(glob.glob(args.veldir + '/' + 'velfltimg*.H'))[filebeg:filebeg+nfiles]

# Find the matching files
resmtch = []; ptbmtch = []; imgmtch = []; fltmtch = []; cnvmtch = []
fctr = 0
for icnv,iflt in zip(cnvfiles,fltfiles):
  # Get the number of files
  faxes,_ = sep.read_file(iflt)
  nex = faxes.n[2]
  fnum = int(iflt.split('velfltlbl')[-1].split('.H')[0])
  # Loop over all examples
  for iex  in range(nex):
    # Get migration files
    ires = args.resdir + '/' + resprefix + '-' + create_inttag(fnum+iex,10000) + '.H'
    iptb = args.ptbdir + '/' + ptbprefix + '-' + create_inttag(fnum+iex,10000) + '.H'
    iimg = args.imgdir + '/' + imgprefix + '-' + create_inttag(fnum+iex,10000) + '.H'
    if(ires in resfiles and iptb in ptbfiles and iimg in imgfiles):
      fltmtch.append(iflt); resmtch.append(ires); imgmtch.append(iimg)
      ptbmtch.append(iptb); cnvmtch.append(icnv)

ntot = len(fltmtch)

# Open H5 files
hfr = h5py.File(outres,'w')
hfi = h5py.File(outimg,'w')
hfp = h5py.File(outptb,'w')
hff = h5py.File(outlbl,'w')
hfc = h5py.File(outcnv,'w')

fctr = 0
# Read in the files and write as HDF5
for ifile in progressbar(range(ntot), "nfiles:"):
  # Residual migration
  raxes,res = sep.read_file(resmtch[ifile])
  [nz,nx,nh,nro] = raxes.n
  res = res.reshape(raxes.n,order='F')
  rwd = res[:,:,16,:]
  # Focused image
  iaxes,img = sep.read_file(imgmtch[ifile])
  img = img.reshape(iaxes.n,order='F')
  iwd = img[:,:,16]
  # Perturbation
  paxes,ptb = sep.read_file(ptbmtch[ifile])
  ptb = ptb.reshape(paxes.n,order='F')
  # Fault label
  faxes,flt = sep.read_file(fltmtch[ifile])
  if(ifile != 0):
    if(fltmtch[ifile] == fltmtch[ifile-1]):
      fctr += 1
    else:
      fctr = 0
  flt = flt.reshape(faxes.n,order='F').T
  # Resample the fault label
  flti = dlut.thresh((dlut.resample(flt[fctr],[nx,nz],kind='linear')),0)
  # Convolution image
  caxes,cnv = sep.read_file(cnvmtch[ifile])
  cnv = cnv.reshape(caxes.n,order='F').T
  cnvi = (dlut.resample(cnv[fctr],[nx,nz],kind='linear'))
  # Write to H5 files
  datatag = create_inttag(ifile,nfiles)
  hfr.create_dataset("res"+datatag, (nro,nx,nz), data=rwd.T, dtype=np.float32)
  hfi.create_dataset("img"+datatag, (nx,nz), data=iwd.T, dtype=np.float32)
  hfp.create_dataset("ptb"+datatag, (nx,nz), data=ptb.T, dtype=np.float32)
  hff.create_dataset("flt"+datatag, (nx,nz), data=flti, dtype=np.float32)
  hfc.create_dataset("cnv"+datatag, (nx,nz), data=cnvi, dtype=np.float32)

# Close H5 files
hfr.close; hfi.close; hfp.close; hff.close(); hfc.close()
