"""
Creates patches of the different image types

@author: Joseph Jennings
@version: 2020.05.17
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import glob
import h5py
from genutils.ptyprint import progressbar, create_inttag
import numpy as np
from resfoc.gain import agc
from deeplearn.utils import plotseglabel, normextract
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from genutils.plot import plot_imgpang, plot_allanggats
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "datdir": "/data/sep/joseph29/projects/resfoc/bench/dat/focdefoc",
    "resdir": "/data/sep/joseph29/projects/resfoc/bench/dat/resdefoc",
    "fprefix": "fag-",
    "dprefix": "dag-",
    "rprefix": "resa-",
    "lprefix": "lbl-",
    "fout": "",
    "dout": "",
    "rout": "",
    "lout": "",
    "ptchz": 64,
    "ptchx": 64,
    "strdz": 32,
    "strdx": 32,
    "norm": 'y',
    "pthresh": 20,
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
ioArgs.add_argument("-datdir",help="Directory containing focused images",type=str,required=True)
ioArgs.add_argument("-resdir",help="Directory residually defocused images",type=str,required=True)
ioArgs.add_argument("-fprefix",help="Prefix to focused angle gathers",type=str,required=True)
ioArgs.add_argument("-lprefix",help="Prefix to fault labels",type=str,required=True)
ioArgs.add_argument("-rprefix",help="Prefix to residually defocused prestack images",type=str,required=True)
ioArgs.add_argument("-fout",help="Output H5 focused file",type=str,required=True)
ioArgs.add_argument("-dout",help="Output H5 defocused file",type=str,required=True)
ioArgs.add_argument("-rout",help="Output H5 residual defocused file",type=str,required=True)
ioArgs.add_argument("-lout",help="Output H5 label file",type=str,required=True)
ioArgs.add_argument("-filebeg",help="Beginning file to use",type=int,required=True)
ioArgs.add_argument("-fileend",help="Ending file to use",type=int,required=True)
# Patching arguments
ptchArgs = parser.add_argument_group('Patching parameters')
ptchArgs.add_argument("-ptchz",help="Size of patch in z [64]",type=int)
ptchArgs.add_argument("-ptchx",help="Size of patch in x [64]",type=int)
ptchArgs.add_argument("-nang",help="Number of angles (all angles included in a patch) [64]",type=int)
ptchArgs.add_argument("-strdz",help="Patch stride (overlap) in z  [32]",type=int)
ptchArgs.add_argument("-strdx",help="Patch stride (overlap) in x [32]",type=int)
# Label arguments
lblArgs = parser.add_argument_group('Label creation parameters')
lblArgs.add_argument("-norm",help="Normalize each patch from -1 to 1 [y]",type=str)
lblArgs.add_argument("-pthresh",help="Number of pixels in a patch to determine if it has a fault",type=int)
# Window arguments
windArgs = parser.add_argument_group('Windowing parameters')
windArgs.add_argument("-fx",help="First x sample [256]",type=int)
windArgs.add_argument("-nxw",help="Length of window in x [512]",type=int)
windArgs.add_argument("-fz",help="First z sample [138]",type=int)
windArgs.add_argument("-nzw",help="Length of window in z [256]",type=int)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
othArgs.add_argument("-qcplot",help="Plot the labels and images (y or [n])",type=str)
othArgs.add_argument("-ptchplot",help="Plot the extracted patches (y or [n])",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Setup IO
sep = seppy.sep()

# Get command line parameters
nzp = args.ptchz; nxp = args.ptchx
strdz = args.strdz; strdx = args.strdx

# Patch extractor for labels
pe = PatchExtractor((nzp,nxp),stride=(strdz,strdx))

# Windowing parameters
fx = args.fx; nxw = args.nxw
fz = args.fz; nzw = args.nzw

# Flags
norm     = sep.yn2zoo(args.norm)
verb     = sep.yn2zoo(args.verb)
qcplot   = sep.yn2zoo(args.qcplot)
ptchplot = sep.yn2zoo(args.ptchplot)

# Get SEPlib files
focfiles = sorted(glob.glob(args.datdir + '/' + args.fprefix + '*.H'))[args.filebeg:args.fileend]
deffiles = sorted(glob.glob(args.datdir + '/' + args.dprefix + '*.H'))[args.filebeg:args.fileend]
lblfiles = sorted(glob.glob(args.datdir + '/' + args.lprefix + '*.H'))[args.filebeg:args.fileend]
resfiles = sorted(glob.glob(args.resdir + '/' + args.rprefix + '*.H'))[args.filebeg:args.fileend]
ntot = len(focfiles)

# Patch parameters
nzp = args.ptchz; strdz = args.strdz
nxp = args.ptchx; strdx = args.strdx
nang = args.nang

# Create H5 File for each type of file
hff = h5py.File(args.fout,'w')
hfd = h5py.File(args.dout,'w')
hfr = h5py.File(args.rout,'w')
hfl = h5py.File(args.lout,'w')

for iex in progressbar(range(ntot), "nfiles:"):
  # Read in and window the focused image
  faxes,fimg = sep.read_file(focfiles[iex])
  fimg   = fimg.reshape(faxes.n,order='F')
  fimgt  = np.ascontiguousarray(fimg.T)[fx:fx+nxw,:,fz:fz+nzw] # [nx,na,nz]
  gfimgt = agc(fimgt.astype('float32'))
  gfimg  = np.ascontiguousarray(np.transpose(gfimgt,(1,2,0))) # [nx,na,nz] -> [na,nz,nx]
  # Read in and window the defocused image
  daxes,dimg = sep.read_file(deffiles[iex])
  dimg    = dimg.reshape(daxes.n,order='F')
  dimgt  = np.ascontiguousarray(dimg.T)[fx:fx+nxw,:,fz:fz+nzw] # [nx,na,nz]
  gdimgt = agc(dimgt.astype('float32'))
  gdimg  = np.ascontiguousarray(np.transpose(gdimgt,(1,2,0))) # [nx,na,nz] -> [na,nz,nx]
  # Read in and window the residually defocused image
  raxes,rimg = sep.read_file(resfiles[iex])
  rimg   = rimg.reshape(raxes.n,order='F')
  rimgt  = np.ascontiguousarray(rimg.T)[fx:fx+nxw,:,fz:fz+nzw] # [nx,na,nz]
  grimgt = agc(rimgt.astype('float32'))
  grimg  = np.ascontiguousarray(np.transpose(grimgt,(1,2,0))) # [nx,na,nz] -> [na,nz,nx]
  # Read in the label
  laxes,lbl = sep.read_file(lblfiles[iex])
  lbl = lbl.reshape(laxes.n,order='F')
  lblw = lbl[fz:fz+nzw,fx:fx+nxw]
  # Extract patches for each input
  fptch = normextract(gfimg,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=True)
  dptch = normextract(gdimg,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=True)
  rptch = normextract(grimg,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=True)
  lptch = normextract(lblw,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=False)
  # Get dimensions
  nptch = fptch.shape[0]
  # Write the data to output each H5 file
  datatag = create_inttag(iex,ntot)
  hff.create_dataset("x"+datatag, (nptch,nang,nzp,nxp,1), data=np.expand_dims(fptch,axis=-1), dtype=np.float32)
  hfd.create_dataset("x"+datatag, (nptch,nang,nzp,nxp,1), data=np.expand_dims(dptch,axis=-1), dtype=np.float32)
  hfr.create_dataset("x"+datatag, (nptch,nang,nzp,nxp,1), data=np.expand_dims(rptch,axis=-1), dtype=np.float32)
  hfl.create_dataset("y"+datatag, (nptch,nzp,nxp,1), data=np.expand_dims(lptch,axis=-1), dtype=np.float32)
  # Plot image and segmentation label
  if(ptchplot):
    iptch = np.random.randint(nptch)
    plot_imgpang(fptch[iptch],0.01,0.01,32,-70.0,2.2222,show=False,vmina=-2.5,vmaxa=2.5,wspace=0.0)
    plot_imgpang(dptch[iptch],0.01,0.01,32,-70.0,2.2222,show=False,vmina=-2.5,vmaxa=2.5,wspace=0.0)
    plot_imgpang(rptch[iptch],0.01,0.01,32,-70.0,2.2222,show=False,vmina=-2.5,vmaxa=2.5,wspace=0.0)
    fig = plt.figure(4,figsize=(8,8)); ax = fig.gca()
    ax.imshow(lptch[iptch],cmap='jet',vmin=0,vmax=1)
    ax.tick_params(labelsize=15)
    plt.show()
  if(qcplot):
    plot_allanggats(gfimg,0.01,0.01,jx=10,transp=True,show=False,vmin=-2.5,vmax=2.5)
    plot_allanggats(gdimg,0.01,0.01,jx=10,transp=True,show=False,vmin=-2.5,vmax=2.5)
    plot_allanggats(grimg,0.01,0.01,jx=10,transp=True,show=False,vmin=-2.5,vmax=2.5)
    gzofimg = np.sum(gfimg,axis=0)
    plotseglabel(gzofimg,lblw,pclip=1.0,show=True)

# Close the H5 files
hff.close(); hfd.close(); hfr.close(); hfl.close()

