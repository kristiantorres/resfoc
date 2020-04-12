"""
Converts prestack images in SEPlib
format to H5 format. Takes only the rho=1 residual migration
image and all of the subsurface offset images

User must specify the directory in which the data
are located and the output .h5 file.

@author: Joseph Jennings
@version: 2020.01.02
"""

import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import subprocess, glob
import deeplearn.utils as dlutil
import h5py

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "sepdir": None,
    "verb": "y",
    "ro1idx": 5,
    "nprint": 100,
    "prefix": "",
    "dsize": 20,
    "savedict": 'y',
    "interp": 'y',
    "norm": 'y',
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
parser.add_argument("-sepdir",help="Directory containing SEPlib files",required=True,type=str)
parser.add_argument("-out",help="output h5 file",required=True,type=str)
# Optional arguments
parser.add_argument("-dsize",help="Number of examples in a single H5 dataset [20]",type=int)
parser.add_argument("-ro1idx",help="Index of rho=1 image [5]",type=int)
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-nprint",help="Print after so many examples [100]",type=int)
parser.add_argument("-prefix",help="Prefix to lbl or img (for example test) [None]",type=str)
parser.add_argument("-savedict",help="Flag if file dictionary should be saved [y]",type=str)
parser.add_argument("-interp",help="Resample the label and features to the nearest power of two",type=str)
parser.add_argument("-norm",help="Normalize the migrated images",type=str)
args = parser.parse_args(remaining_argv)

# Set up IO
sep = seppy.sep(sys.argv)

# Get command line arguments
sepdir = args.sepdir; out = args.out
ro1idx = args.ro1idx

verb = sep.yn2zoo(args.verb); nprint = args.nprint
savedict = sep.yn2zoo(args.savedict)

interp = sep.yn2zoo(args.interp); norm = sep.yn2zoo(args.norm)

prefix = args.prefix
dsize = args.dsize

# First get total number of examples available
imgfiles = sorted(glob.glob(sepdir + '/' + prefix + 'img*.H'))
lblfiles = sorted(glob.glob(sepdir + '/' + prefix + 'lbl*.H'))

assert(len(imgfiles) == len(lblfiles)), "Must have as many features as labels. Exiting"

ntot = 0
for ifile in imgfiles: ntot += (sep.read_header(None,ifname=ifile)).n[4]

if(verb): print("Total number of examples: %d"%(ntot))

assert(dsize <= ntot), "Batch size must be <= total number of examples"

# Calculate the nearest number lt than ntot that is n%dsize = 0
nex = ntot - ntot%dsize
if(verb): print("Using %d examples"%(nex))

# Input axes
iaxes = sep.read_header(None,ifname=imgfiles[0])
nzc = iaxes.n[0]; nxc = iaxes.n[1]; nh = iaxes.n[2]

# Output interpolated axes
nzf = None; nxf = None
if(interp):
  nzf = dlutil.next_power_of_2(nzc)
  nxf = dlutil.next_power_of_2(nxc)
else:
  nzf = nzc; nxf = nxc

# Allocate output batch arrays
imgb = np.zeros([nzf,nxf,nh,dsize],dtype='float32')
rhob = np.zeros([nzf,nxf,dsize],dtype='float32')

nwrt = 0; k = 0; fctr = 0
xs = []; ys = []; fs = []; fdict = {}
with h5py.File(out,'w') as hf:
  while nwrt < nex:
    imgb[:] = 0.0; rhob[:] = 0.0
    if(verb):
      if(nwrt%nprint == 0):
        print("%d ... "%(nwrt),end=''); sys.stdout.flush()
    # Read in image
    iaxes,img = sep.read_file(None,ifname=imgfiles[fctr])
    img = img.reshape(iaxes.n,order='F')[:,:,:,ro1idx,:]
    if(interp):
      img = (dlutil.resizepow2(img.T)).T
    if(norm):
      img = dlutil.normalize(img)
    # Read in rho
    raxes,rho = sep.read_file(None,ifname=lblfiles[fctr])
    rho = rho.reshape(raxes.n,order='F')
    if(interp):
      rho = (dlutil.resizepow2(rho.T)).T
    k += iaxes.n[4]
    xs.append(img); ys.append(rho)
    fs.append((imgfiles[fctr],lblfiles[fctr]))
    # If reached dsize examples, save to H5 file
    if(k >= dsize):
      nfit = k//dsize
      beg = 0; end = dsize
      for ifit in range(nfit):
        # Save to output arrays
        catx = np.concatenate(xs,axis=3)
        caty = np.concatenate(ys,axis=2)
        imgb[:,:,:,:] = catx[:,:,:,beg:end]
        rhob[:,:,:] = caty[:,:,beg:end]
        imgbt = np.transpose(imgb,(3,0,1,2))
        rhobt = np.transpose(rhob,(2,0,1))
        datatag = sep.create_inttag(nwrt,nex)
        # Save to file
        hf.create_dataset("x"+datatag, (dsize,nzf,nxf,nh), data=imgbt, dtype=np.float32)
        hf.create_dataset("y"+datatag, (dsize,nzf,nxf), data=rhobt, dtype=np.float32)
        # Residual is used after first iteration
        if(ifit == 1 and len(fs) > 1): del fs[0]
        fdict["f"+datatag] = fs.copy()
        k = catx.shape[3] - end
        nwrt += dsize; beg = end; end += dsize
      # Save residuals
      xs = []; ys = []
      fs = []
      if(k != 0):
        xs.append(catx[:,:,:,end-dsize:]); ys.append(caty[:,:,end-dsize:])
        fs.append((imgfiles[fctr],lblfiles[fctr]))
    # Increase the file counter
    fctr += 1
print(" ")

if(savedict):
  bname,_ = os.path.splitext(out)
  np.save(bname+'.npy',fdict)
