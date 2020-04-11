"""
Converts the prestack residual migration images from SEPlib
format to H5 format. User chooses whether to keep all
axes, the residual migration images, the subsurface offsets
or just the migrated images themselves

User must specify the directory in which the data
are located and the output .h5 file.

@author: Joseph Jennings
@version: 2020.01.07
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
    "zidx": 10,
    "ro1idx": 5,
    "nprint": 100,
    "prefix": "",
    "dsize": 20,
    "savedict": 'y',
    "pow2": 'y',
    "norm": 'y',
    "keepres": 'y',
    "keepoff": 'n',
    "ntot": 0,
    "nxo": None,
    "nzo": None,
    "split": 0.8,
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
ioArgs = parser.add_argument_group("Input output arguments")
ioArgs.add_argument("-sepdir",help="Directory containing SEPlib files",type=str)
ioArgs.add_argument("-trout",help="Output training H5 file",type=str)
ioArgs.add_argument("-vaout",help="Output validation H5 file",type=str)
# Optional arguments
parser.add_argument("-ntot",help="Total number of examples to write [all files]",type=int)
parser.add_argument("-dsize",help="Number of examples in a single H5 dataset [20]",type=int)
parser.add_argument("-zidx",help="Index of zero subsurface offset [10]",type=int)
parser.add_argument("-ro1idx",help="Index of rho=1 image [5]",type=int)
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-nprint",help="Print after so many examples [100]",type=int)
parser.add_argument("-prefix",help="Prefix to lbl or img (for example test) [None]",type=str)
parser.add_argument("-savedict",help="Flag if file dictionary should be saved [y]",type=str)
parser.add_argument("-pow2",help="Resample the label and features to the nearest power of two",type=str)
parser.add_argument("-nzo",help="Interpolate the image so it has nzo depth samples [nz]",type=int)
parser.add_argument("-nxo",help="Interpolate the image so it has nxo lateral samples [nx]",type=int)
parser.add_argument("-norm",help="Normalize the computed image [y]",type=str)
parser.add_argument("-keepres",help="Keep the residual migration images as part of the training [y]",type=str)
parser.add_argument("-keepoff",help="Keep the subsurface offsets as part of the training [n]",type=str)
parser.add_argument("-split",help="Percentage split between training and validation [0.8]",type=float)
args = parser.parse_args(remaining_argv)

# Set up IO
sep = seppy.sep(sys.argv)

# Get command line arguments
sepdir = args.sepdir
trout = args.trout; vaout = args.vaout
split = args.split
prefix = args.prefix

keepoff = sep.yn2zoo(args.keepoff)
keepres = sep.yn2zoo(args.keepres)
zidx = args.zidx; ro1idx = args.ro1idx
assert((keepoff == True and keepres == True) == False), """Keeping both offsets
and residual migrations is not supported currently"""

verb = sep.yn2zoo(args.verb); nprint = args.nprint
savedict = sep.yn2zoo(args.savedict)

pow2 = sep.yn2zoo(args.pow2); norm = sep.yn2zoo(args.norm)
dsize = args.dsize

# First get total number of examples available
imgfiles = sorted(glob.glob(sepdir + '/' + prefix + 'img*.H'))
lblfiles = sorted(glob.glob(sepdir + '/' + prefix + 'lbl*.H'))

assert(len(imgfiles) == len(lblfiles)), "Must have as many features as labels. Exiting"

ntot = args.ntot
if(ntot == 0):
  for ifile in imgfiles: ntot += (sep.read_header(None,ifname=ifile)).n[4]

if(verb): print("Total number of examples: %d"%(ntot))

assert(dsize <= ntot), "Batch size must be <= total number of examples"

# Calculate the nearest number lt than ntot that is n%dsize = 0
nex  = ntot - ntot%dsize
nb   = nex//dsize
nbtr = int(nb*split)
nbva = nb - nbtr
if(verb):
  print("Using %d examples"%(nex))
  print("Total number of batches: %d"%(nb))
  print("Training batches: %d"%(nbtr))
  print("Validation batches: %d"%(nbva))

# Input axes
iaxes = sep.read_header(None,ifname=imgfiles[0])
nzc = iaxes.n[0]; nxc = iaxes.n[1]; nh = iaxes.n[2]; nro = iaxes.n[3]

# Output interpolated axes
nzf = args.nzo; nxf = args.nxo
interp = True
if(pow2):
  nzf = dlutil.next_power_of_2(nzc)
  nxf = dlutil.next_power_of_2(nxc)
elif(nzf == None and nxf == None):
  nzf = nzc; nxf = nxc
  interp = False

# Allocate output batch arrays
if(keepoff):
  imgb = np.zeros([nzf,nxf,nh,dsize],dtype='float32')
  rhob = np.zeros([nzf,nxf,dsize],dtype='float32')
elif(keepres):
  imgb = np.zeros([nzf,nxf,nro,dsize],dtype='float32')
  rhob = np.zeros([nzf,nxf,dsize],dtype='float32')
else:
  imgb = np.zeros([nzf,nxf,dsize],dtype='float32')
  rhob = np.zeros([nzf,nxf,dsize],dtype='float32')

nwrt = 0; nwrb = 0; k = 0; fctr = 0
xs = []; ys = []; fs = []; fdict = {}
# Open H5 files
hftr = h5py.File(trout,'w')
hfva = h5py.File(vaout,'w')
# Loop over all examples
while nwrt < nex:
  imgb[:] = 0.0; rhob[:] = 0.0
  if(verb):
    if(nwrt%nprint == 0):
      print("%d ... "%(nwrt),end=''); sys.stdout.flush()
  # Read in image
  iaxes,img = sep.read_file(None,ifname=imgfiles[fctr])
  if(keepoff):
    img = img.reshape(iaxes.n,order='F')[:,:,:,roi1dx,:]
  elif(keepres):
    img = img.reshape(iaxes.n,order='F')[:,:,zidx,:,:]
  else:
    img = img.reshape(iaxes.n,order='F')[:,:,zidx,ro1idx,:]
  # Process the image
  if(interp):
    img = (dlutil.resample(img.T,[nxf,nzf])).T
  if(norm):
    img = dlutil.normalize(img)
  # Read in rho
  raxes,rho = sep.read_file(None,ifname=lblfiles[fctr])
  rho = rho.reshape(raxes.n,order='F')
  if(interp):
    rho = (dlutil.resample(rho.T,[nxf,nzf])).T
  k += iaxes.n[4]
  xs.append(img); ys.append(rho)
  fs.append((imgfiles[fctr],lblfiles[fctr]))
  # If reached dsize examples, save to H5 file
  if(k >= dsize):
    nfit = k//dsize
    beg = 0; end = dsize
    for ifit in range(nfit):
      # First write feature
      datatag = sep.create_inttag(nwrt,nex)
      if(keepoff or keepres):
        catx = np.concatenate(xs,axis=3)
        imgb[:,:,:,:] = catx[:,:,:,beg:end]
        imgbt = np.transpose(imgb,(3,0,1,2))
        # Write to the training file
        if(nwrb < nbtr):
          if(keepoff):
            hftr.create_dataset("x"+datatag, (dsize,nzf,nxf,nh), data=imgbt, dtype=np.float32)
          elif(keepres):
            hftr.create_dataset("x"+datatag, (dsize,nzf,nxf,nro), data=imgbt, dtype=np.float32)
        # Write to the validation file
        else:
          if(keepoff):
            hfva.create_dataset("x"+datatag, (dsize,nzf,nxf,nh), data=imgbt, dtype=np.float32)
          elif(keepres):
            hfva.create_dataset("x"+datatag, (dsize,nzf,nxf,nro), data=imgbt, dtype=np.float32)
        k = catx.shape[3] - end
      else:
        catx = np.concatenate(xs,axis=2)
        imgb[:,:,:,:] = catx[:,:,beg:end]
        imgbt = np.transpose(imgb,(2,0,1))
        if(nwrb < nbtr):
          hftr.create_dataset("x"+datatag, (dsize,nzf,nxf), data=imgbt, dtype=np.float32)
        else:
          hftr.create_dataset("x"+datatag, (dsize,nzf,nxf), data=imgbt, dtype=np.float32)
        k = catx.shape[2] - end
      # Now write label
      caty = np.concatenate(ys,axis=2)
      rhob[:,:,:] = caty[:,:,beg:end]
      rhobt = np.transpose(rhob,(2,0,1))
      # Save to file
      if(nwrb < nbtr):
        hftr.create_dataset("y"+datatag, (dsize,nzf,nxf), data=rhobt, dtype=np.float32)
      else:
        hfva.create_dataset("y"+datatag, (dsize,nzf,nxf), data=rhobt, dtype=np.float32)
      # Residual is used after first iteration
      if(ifit == 1 and len(fs) > 1): del fs[0]
      fdict["f"+datatag] = fs.copy()
      # Increment counters
      nwrt += dsize; nwrb += 1; beg = end; end += dsize
    # Save residuals
    xs = []; ys = []
    fs = []
    if(k != 0):
      if(keepoff or keepres):
        xs.append(catx[:,:,:,end-dsize:]); ys.append(caty[:,:,end-dsize:])
      else:
        xs.append(catx[:,:,end-dsize:]); ys.append(caty[:,:,end-dsize:])
      fs.append((imgfiles[fctr],lblfiles[fctr]))
  # Increase the file counter
  fctr += 1
print(" ")
# Close the H5files
hftr.close(); hfva.close()

if(savedict):
  bname,_ = os.path.splitext(trout)
  np.save(bname+'.npy',fdict)

