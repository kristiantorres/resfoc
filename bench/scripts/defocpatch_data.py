"""
Labels patches as defocused or focused

@author: Joseph Jennings
@version: 2020.05.17
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import h5py
from utils.ptyprint import progressbar, create_inttag
import numpy as np
from deeplearn.utils import plotseglabel, plotsegprobs, normalize
from deeplearn.dataloader import load_all_unlabeled_data,load_unlabeled_flat_data
from deeplearn.focuslabels import corrsim
import random
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "focptch": "",
    "defptch": "",
    "focprb": "" ,
    "defprb": "",
    "begex": 0,
    "endex": -1,
    "nqc": 100,
    "thresh1": 0.7,
    "thresh2": 0.5,
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
ioArgs.add_argument("-focptch",help="H5 file containing focused patches",type=str,required=True)
ioArgs.add_argument("-defptch",help="H5 file containing defocused patches",type=str,required=True)
ioArgs.add_argument("-focprb",help="H5 file containing focused fault probabilities",type=str,required=True)
ioArgs.add_argument("-defprb",help="H5 file containing defocused fault probabilities",type=str,required=True)
ioArgs.add_argument("-begex",help="Beginning example to process [0]",type=int)
ioArgs.add_argument("-endex",help="Ending example to process [last example]",type=int)
ioArgs.add_argument("-deflbls",help="Output H5 file containing labeled defocused patches",type=str,required=True)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-thresh1",help="Threshold for determining if image is defocused [0.7]",type=float)
othArgs.add_argument("-thresh2",help="Threshold for determining if image is defocused [0.5]",type=float)
othArgs.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
othArgs.add_argument("-qcplot",help="Plot focused and defocuse probablilities (y or [n])",type=str)
othArgs.add_argument("-nqc",help="Number of focused and defocused patches to QC",type=int)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Setup IO
sep = seppy.sep()

# Flags
verb =  sep.yn2zoo(args.verb)

# Read in patch data
focdat = load_all_unlabeled_data(args.focptch)
defdat = load_all_unlabeled_data(args.defptch)

focprb = load_unlabeled_flat_data(args.focprb)
defprb = load_unlabeled_flat_data(args.defprb)

# Get data shape
nex = focdat.shape[0]; nzp = focdat.shape[1]; nxp = focdat.shape[2]
if(verb): print("Total number of examples in a file: %d"%(nex))

# QC the predictions
for iex in progressbar(range(args.nqc),"nqc:"):
  idx = np.random.randint(nex)
  corrimg = corrsim(defdat[idx,:,:,0],focdat[idx,:,:,0])
  corrprb = corrsim(defprb[idx,:,:,0],focprb[idx,:,:,0])
  #plotsegprobs(focdat[idx,:,:,0],focprb[idx,:,:,0],interp='sinc',show=False,pmin=0.2)
  #plotsegprobs(defdat[idx,:,:,0],defprb[idx,:,:,0],interp='sinc',show=False,pmin=0.2)
  #plotseglabel(focdat[idx,:,:,0],fltdat[idx,:,:,0],interp='sinc',show=False)
  #fig,axarr = plt.subplots(1,2,figsize=(10,6))
  #axarr[0].imshow(focdat[idx,:,:,0],cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
  #axarr[0].set_title('idx=%d CORRIMG=%f'%(idx,corrimg),fontsize=15)
  #axarr[1].imshow(defdat[idx,:,:,0],cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
  #axarr[1].set_title('CORRPRB=%f'%(corrprb),fontsize=15)
  #plt.show()

# Output list of defocused images
defout = []

print("Starting at %d and ending at %d"%(args.begex,args.endex))
for iex in progressbar(range(args.begex,args.endex), "nsim:"):
  corrimg = corrsim(defdat[iex,:,:,0],focdat[iex,:,:,0])
  corrprb = corrsim(defprb[iex,:,:,0],focprb[iex,:,:,0])
  if(corrimg < args.thresh1 and corrprb < args.thresh1):
    defout.append(defdat[iex,:,:,0])
  elif(corrimg < args.thresh2 or corrprb < args.thresh2):
    defout.append(defdat[iex,:,:,0])

print("Keeping def=%d examples"%(len(defout)))

# Convert to numpy array
defs = np.asarray(defout)

ntot = defs.shape[0]

# Write the labeled data to file
hfd = h5py.File(args.deflbls,'w')

for iex in progressbar(range(ntot), "iex:"):
  datatag = create_inttag(iex,nex)
  hfd.create_dataset("x"+datatag, (nxp,nzp,1), data=np.expand_dims(defs[iex],axis=-1), dtype=np.float32)
  hfd.create_dataset("y"+datatag, (1,), data=0, dtype=np.float32)

# Close file
hfd.close()

print("Success!")

