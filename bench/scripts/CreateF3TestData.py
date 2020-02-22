"""
Creates patched images from F3 seismic cube

@author: Joseph Jennings
@version: 2020.02.18
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import segyio
import h5py
import numpy as np
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
import deeplearn.utils as dlut
from utils.ptyprint import create_inttag, progressbar
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    "ny": 500,
    "nto": 256,
    "nxo": 1024,
    "ptchz": 128,
    "ptchx": 128,
    "strdz": 64,
    "strdx": 64,
    "norm": 'y',
    "wrtcube": 'y',
    "fs": 0,
    "gridqc": 'y',
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
ioArgs.add_argument("-sgyin",help="Input segyfile",type=str)
ioArgs.add_argument("-h5out",help="Output H5 file",type=str)
# Optional arguments
parser.add_argument("-fs",help="First time sample from which to window the data [0]",type=int)
parser.add_argument("-ny",help="Y dimension of output 3D cube [500]",type=int)
parser.add_argument("-nto",help="Interpolate the entire image before patch extraction [512]",type=int)
parser.add_argument("-nxo",help="Interpolate the entire image before patch extraction [1024]",type=int)
parser.add_argument("-ptchz",help="Size of patch in z [128]",type=int)
parser.add_argument("-ptchx",help="Size of patch in x [128]",type=int)
parser.add_argument("-strdz",help="Patch stride (overlap) in z  [64]",type=int)
parser.add_argument("-strdx",help="Patch stride (overlap) in x [64]",type=int)
parser.add_argument("-norm",help="Normalize each patch from -1 to 1 [y]",type=str)
parser.add_argument("-wrtcube",help="Write out the cube built from the segy [y]",type=str)
parser.add_argument("-gridqc",help="Plot patch grid on top of image [y]",type=str)
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
wrtcub = sep.yn2zoo(args.wrtcube)
gridqc  = sep.yn2zoo(args.gridqc)

# Read in the segy
with segyio.open(args.sgyin,ignore_geometry=True) as f: 
  data = f.trace.raw[:]
  dt = f.attributes(segyio.TraceField.TRACE_SAMPLE_INTERVAL)[0][0]/1e6
  dx = 25; dy = 25
  #dx = (f.attributes(segyio.TraceField.CDP_X)[1][0] - f.attributes(segyio.TraceField.CDP_X)[0][0])/1000.0
  #dy = (f.attributes(segyio.TraceField.CDP_Y)[1][0] - f.attributes(segyio.TraceField.CDP_Y)[0][0])/1000.0

# Reshape the data so it is a regular cube
fs = args.fs
nt = data.shape[1] - fs; nx = 951; ny = args.ny
ntr = nx*ny
datawind = data[:ntr,fs:]
datawind = datawind.reshape([ny,nx,nt])

# Patch the interpolated slices
nzp = args.ptchz; nxp = args.ptchx
pshape = (nzp,nxp)
pstride = (args.strdz,args.strdx)
pe = PatchExtractor(pshape,stride=pstride)

# Open the H5 file
hf = h5py.File(args.h5out,'w')

# Extract the desired slices, interpolate and patch
nxo = args.nxo; nto = args.nto
dout = np.zeros([nto,nxo,ny])
k = 0
for iy in progressbar(range(ny), "xline"):
  dout[:,:,k] = (dlut.resample(datawind[iy,:,:],[nxo,nto],kind='cubic')).T
  # Plot grid on top of interpolated image
  if(gridqc):
    fsize = 18
    xticks = np.arange(0,(nxo+1)*dx/1000.0,nxp*dx/1000.0)
    yticks = np.arange(0,(nto+1)*dt,nzp*dt)
    fig = plt.figure(1,figsize=(10,7)); ax = fig.gca()
    plt.imshow(dout[:,:,k],cmap='gray',extent=[0.0,(nxo-1)*dx/1000.0,(nto-1)*dt,0.0],vmin=-7000,vmax=7000,interpolation='sinc')
    plt.rc('grid', linestyle="-", color='black')
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    ax.set_xlabel('Xline (km)',fontsize=fsize)
    ax.set_ylabel('Time (s)',fontsize=fsize)
    ax.tick_params(labelsize=fsize)
    ax.set_aspect(6.5)
    plt.grid()
    plt.show()
  # Extract the patches
  iptch = pe.extract(dout[:,:,k])
  # Flatten
  pz = iptch.shape[0]; px = iptch.shape[1]
  iptch = iptch.reshape([pz*px,nzp,nxp])
  niptch = np.zeros(iptch.shape)
  for ip in range(pz*px):
    niptch[ip,:,:] = dlut.normalize(iptch[ip,:,:])
  datatag = create_inttag(k,ny)
  # Save to dataset
  hf.create_dataset("x"+datatag, (pz*px,nzp,nxp,1), data=np.expand_dims(niptch,axis=-1), dtype=np.float32)
  k += 1

# Close the H5 file
hf.close()

# Write out the cube if desired
if(wrtcub):
  axes = seppy.axes([nt,nx,ny],[0.0,0.0,0.0],[dt,dx,dy])
  sep.write_file(None,axes,datawind.T,ofname='f3cube.H')

