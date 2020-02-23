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
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "sepdir": None,
    "verb": "y",
    "prefix": "",
    "nzo": 512,
    "nxo": 1024,
    "norm": 'y',
    "ptchz": 128,
    "ptchx": 128,
    "nfiles": 210,
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
parser.add_argument("-sepdir",help="Directory containing SEPlib files",type=str)
parser.add_argument("-out",help="output h5 file",type=str)
# Optional arguments
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-prefix",help="Prefix to lbl or img (for example test) [None]",type=str)
parser.add_argument("-nzo",help="Interpolate the entire image before patch extraction [512]",type=int)
parser.add_argument("-nxo",help="Interpolate the entire image before patch extraction [1024]",type=int)
parser.add_argument("-ptchz",help="Size of patch in z [128]",type=int)
parser.add_argument("-ptchx",help="Size of patch in x [128]",type=int)
parser.add_argument("-strdz",help="Patch stride (overlap) in z  [64]",type=int)
parser.add_argument("-strdx",help="Patch stride (overlap) in x [64]",type=int)
parser.add_argument("-norm",help="Normalize each patch from -1 to 1 [y]",type=str)
parser.add_argument("-nfiles",help="Number of SEPlib files to use [210]",type=int)
args = parser.parse_args(remaining_argv)

# Set up IO
sep = seppy.sep(sys.argv)

# Get command line arguments
sepdir = args.sepdir; out = args.out
prefix = args.prefix
nfiles = args.nfiles

norm = sep.yn2zoo(args.norm)

verb = sep.yn2zoo(args.verb)

# Get SEPlib files
imgfiles = sorted(glob.glob(sepdir + '/' + prefix + 'img*.H'))[:nfiles]
lblfiles = sorted(glob.glob(sepdir + '/' + prefix + 'lbl*.H'))[:nfiles]
#imgfiles = sorted(glob.glob(sepdir + '/' + prefix + 'img*.H'))[300:nfiles+300]
#lblfiles = sorted(glob.glob(sepdir + '/' + prefix + 'lbl*.H'))[300:nfiles+300]
nofiles = len(imgfiles)

assert(len(imgfiles) == len(lblfiles)), "Must have as many features as labels. Exiting."

ntot = 0
for ifile in imgfiles: ntot += (sep.read_header(None,ifname=ifile)).n[2]

if(verb): print("Total number of examples: %d"%(ntot))

# Input axes
iaxes = sep.read_header(None,ifname=imgfiles[0])
nzi = iaxes.n[0]; nxi = iaxes.n[1]

# Output interpolated size
nzo = args.nzo; nxo = args.nxo

# Output axes
nzp = args.ptchz; nxp = args.ptchx
pshape = (nzp,nxp)
pstride = (args.strdz,args.strdx)
pe = PatchExtractor(pshape,stride=pstride)

# Get pointer to H5 file
hf = h5py.File(out,'w')

k = 0
# Loop over all files
for ifile in progressbar(range(nofiles), "nfiles", 40):
  # Read in image
  iaxes,img = sep.read_file(None,ifname=imgfiles[ifile])
  img = img.reshape(iaxes.n,order='F')
  # Read in label
  laxes,lbl = sep.read_file(None,ifname=lblfiles[ifile])
  lbl = lbl.reshape(laxes.n,order='F')
  # Get total number of images in file
  nm = img.shape[2]
  # Loop over all images in file
  for im in range(nm):
    iimg = img[:,:,im].T
    ilbl = lbl[:,:,im].T
    # Interpolate to patching-friendly sizes
    iimg = (dlut.resample(iimg,[nxo,nzo],kind='linear')).T
    ilbl = (dlut.thresh(dlut.resample(ilbl,[nxo,nzo],kind='linear'),0)).T
    # Extract patches
    iptch = pe.extract(iimg)
    lptch = pe.extract(ilbl)
    # Flatten
    pz = iptch.shape[0]; px = iptch.shape[1]
    iptch = iptch.reshape([pz*px,nzp,nxp])
    lptch = lptch.reshape([pz*px,nzp,nxp])
    niptch = np.zeros(iptch.shape)
    # Normalize the images
    for ip in range(pz*px):
      niptch[ip,:,:] = dlut.normalize(iptch[ip,:,:])
    # Save to dataset
    datatag = create_inttag(k,ntot)
    hf.create_dataset("x"+datatag, (pz*px,nzp,nxp,1), data=np.expand_dims(niptch,axis=-1), dtype=np.float32)
    hf.create_dataset("y"+datatag, (pz*px,nzp,nxp,1), data=np.expand_dims( lptch,axis=-1), dtype=np.float32)
    k += 1

hf.close()

