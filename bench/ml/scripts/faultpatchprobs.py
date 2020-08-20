"""
Generates fault probabilities on migrated image patches

@author: Joseph Jennings
@version: 2020.05.17
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import h5py
from genutils.ptyprint import progressbar, create_inttag
import numpy as np
from deeplearn.utils import plotseglabel, plotsegprobs, normalize
from deeplearn.dataloader import load_all_unlabeled_data,load_unlabeled_flat_data
from deeplearn.focuslabels import corrsim
import tensorflow as tf
from tensorflow.keras.models import model_from_json
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
    "lblptch": "",
    "wgts": "",
    "arch": "",
    "focprb": "" ,
    "defprb": "",
    "gpus": [],
    "nqc": 100,
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
ioArgs.add_argument("-lblptch",help="H5 file containing fault labels",type=str,required=True)
ioArgs.add_argument("-wgts",help="CNN weights",type=str,required=True)
ioArgs.add_argument("-arch",help="CNN architecture",type=str,required=True)
ioArgs.add_argument("-focprb",help="Output H5 focused fault probabilities",type=str,required=True)
ioArgs.add_argument("-defprb",help="Output H5 defocused fault probabilities",type=str,required=True)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-thresh",help="Threshold for determining if image is defocused",type=float)
othArgs.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
othArgs.add_argument("-qcplot",help="Plot focused and defocuse probablilities (y or [n])",type=str)
othArgs.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
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

# Get GPU to use
gpus  = sep.read_list(args.gpus,[])
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)

# Read in patch data
focdat = load_all_unlabeled_data(args.focptch)
defdat = load_all_unlabeled_data(args.defptch)
#fltdat = load_all_unlabeled_data(args.lblptch)

# Get data shape
nex = focdat.shape[0]; nzp = focdat.shape[1]; nxp = focdat.shape[2]
if(verb): print("Total number of examples in a file: %d"%(nex))

# Read in CNN architecture and weights
with open(args.arch,'r') as f:
  model = model_from_json(f.read())

wgts = model.load_weights(args.wgts)

if(verb): model.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Predict on each dataset
focprb = model.predict(focdat,verbose=1)
defprb = model.predict(defdat,verbose=1)

# Write fault probabilities to file
hff = h5py.File(args.focprb,'w')
hfd = h5py.File(args.defprb,'w')

for iex in progressbar(range(nex),"nex:"):
  datatag = create_inttag(iex,nex)
  hff.create_dataset("x"+datatag, (nxp,nzp,1), data=focprb[iex], dtype=np.float32)
  hfd.create_dataset("x"+datatag, (nxp,nzp,1), data=defprb[iex], dtype=np.float32)

hff.close(); hfd.close();

# QC the predictions
for iex in progressbar(range(args.nqc),"nqc:"):
  idx = np.random.randint(nex)
  corrimg = corrsim(defdat[idx,:,:,0],focdat[idx,:,:,0])
  corrprb = corrsim(defprb[idx,:,:,0],focprb[idx,:,:,0])
  plotsegprobs(focdat[idx,:,:,0],focprb[idx,:,:,0],interp='sinc',show=False,pmin=0.2)
  plotsegprobs(defdat[idx,:,:,0],defprb[idx,:,:,0],interp='sinc',show=False,pmin=0.2)
  #plotseglabel(focdat[idx,:,:,0],fltdat[idx,:,:,0],interp='sinc',show=False)
  fig,axarr = plt.subplots(1,2,figsize=(10,6))
  axarr[0].imshow(focdat[idx,:,:,0],cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
  axarr[0].set_title('idx=%d CORRIMG=%f'%(idx,corrimg),fontsize=15)
  axarr[1].imshow(defdat[idx,:,:,0],cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
  axarr[1].set_title('CORRPRB=%f'%(corrprb),fontsize=15)
  plt.show()

# Make list copy of the well-focused images
#focout = list(np.copy(focdat))

# Generate long list of random indices
#idxs = random.sample(range(nex), nex)

# Remove the well-focused defocused images
#defout = []

#for iex in progressbar(range(nex), "nsim:"):
#  corrimg = corrsim(defdat[idx,:,:,0],focdat[idx,:,:,0])
#  corrprb = corrsim(defprb[idx,:,:,0],focprb[idx,:,:,0])
#  if(corrimg < args.thresh and corrprb < args.thresh):
#    defout.append(defdat[idx,:,:,0])
#  else:
#    del focout[idxs[iex]]

#print("Keeping def=%d foc=%d examples"%(len(defout),len(focout)))

# Convert to numpy arrays
#defs = np.asarray(defout); focs = np.asarray(focout)

#ntot = defs.shape[0]

# Write the labeled data to file
#hff = h5py.File(args.focout,'w')
#hfd = h5py.File(args.defout,'w')

#for iex in progressbar(range(ntot), "iex:"):
#  datatag = create_inttag(iex,ntot)
#  hff.create_dataset("x"+datatag, (nxp,nzp,1), data=np.exand_dims(focs[iex],axis=-1), dtype=np.float32)
#  hff.create_dataset("y"+datatag, (1,), data=1, dtype=np.float32)
#  hfd.create_dataset("x"+datatag, (nxp,nzp,1), data=np.expand_dims(defs[iex],axis=-1), dtype=np.float32)
#  hfd.create_dataset("y"+datatag, (1,), data=0, dtype=np.float32)

# Close files
#hff.close(); hfd.close()

