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
from resfoc.estro import estro_tgt, refocusimg
from resfoc.gain import agc
from scaas.trismooth import smooth
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
import matplotlib.pyplot as plt
from utils.movie import viewimgframeskey

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
imgprefix = args.imgprefix
resprefix = args.resprefix
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

if(verb): print("Total number of files: %d"%len(imgmtch))

# Window to the appropriate number of files
imgwind = imgmtch[:nfiles]
reswind = resmtch[:nfiles]
ptbwind = ptbmtch[:nfiles]

# Get the rho axis
raxes,_ = sep.read_file(resmtch[0])
[nz,nx,nh,nro] = raxes.n; [dz,dx,dh,dro] = raxes.d; [oz,ox,oh,oro] = raxes.o

# Build the patch extractor
nzp = args.ptchz; nxp = args.ptchx
pshape = (nro,nxp,nzp)
strdx = args.strdx; strdz = args.strdz
pstride = (nro,strdx,strdz)
pe = PatchExtractor(pshape,stride=pstride)

# Get pointer to H5 file
hf = h5py.File(out,'w')

k = 0
# Loop over the files
for ifile in progressbar(range(nfiles), "nfiles"):
  # Read in the files
  iaxes,img = sep.read_file(imgwind[ifile])
  img = img.reshape(iaxes.n,order='F')
  raxes,res = sep.read_file(reswind[ifile])
  res = res.reshape(raxes.n,order='F')
  paxes,ptb = sep.read_file(ptbwind[ifile])
  ptb = ptb.reshape(paxes.n,order='F')
  #  Window the data and get the rho axis
  #izro = img[nzp:,:,16]; rzro = res[nzp:,:,16,:]; pzro = ptb[nzp:,:]
  izro = img[:,:,16]; rzro = res[:,:,16,:]; pzro = ptb[:,:]
  # Compute the label
  rho,lbls = estro_tgt(rzro.T,izro.T,dro,oro,nzp=nzp,nxp=nxp,strdx=strdx,strdz=strdz,onehot=True)

  # Smooth rho and refocus
  #rhosm = smooth(rho.astype('float32'),rect1=30,rect2=30)
  #ref = refocusimg(rzro.T,rhosm,dro)
  #fig,ax = plt.subplots(1,2,figsize=(14,7))
  #ax[0].imshow(rho.T,cmap='seismic',vmin=0.98,vmax=1.02)
  #ax[1].imshow(-pzro,cmap='seismic',vmin=-100,vmax=100)
  #viewimgframeskey(agc(rzro.T),show=False,pclip=0.6,ttlstring=r'$\rho=%.3f$',dttl=dro,ottl=oro,wbox=10,hbox=6)
  #plt.figure(3,figsize=(14,7))
  #grzro = agc(rzro[:,:,9].astype('float32').T)
  #pclip = 0.8
  #vmin = pclip*np.min(grzro); vmax = pclip*np.max(grzro)
  #plt.imshow(agc(ref).T,cmap='gray',vmin=vmin,vmax=vmax)
  #plt.figure(4,figsize=(14,7))
  #plt.imshow(grzro.T,cmap='gray',vmin=vmin,vmax=vmax)
  #plt.show()

  # Create image patches
  rzrop = np.squeeze(pe.extract(rzro.T))
  # Flatten the labels and write to H5 file
  px = lbls.shape[0]; pz = lbls.shape[1]
  lbls = lbls.reshape([px*pz,nro]); rzrop = rzrop.reshape([px*pz,nro,nxp,nzp])
  rzropt = np.transpose(rzrop,(0,2,3,1))
  datatag = create_inttag(k,nfiles)
  hf.create_dataset("x"+datatag, (pz*px,nxp,nzp,nro), data=rzropt, dtype=np.float32)
  hf.create_dataset("y"+datatag, (pz*px,nro), data=lbls, dtype=np.float32)
  #TODO: save the velocity perturbation files
  #TODO: save the well focused images
  k += 1

hf.close()
