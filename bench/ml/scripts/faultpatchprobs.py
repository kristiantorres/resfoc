"""
Generates fault probabilities on migrated image patches

@author: Joseph Jennings
@version: 2020.05.17
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import h5py
from utils.ptyprint import progressbar, create_inttag
import numpy as np
from deeplearn.utils import plotseglabel, normalize
from deeplearn.dataloader import load_all_unlabeled_data
import tensorflow as tf
from tensorflow.keras.models import model_from_json
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
    "gpus": []
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
ioArgs.add_argument("-focprb",help="Output H5 focused fault probablilities",type=str,required=True)
ioArgs.add_argument("-defprb",help="Output H5 defocused fault probbilities",type=str,required=True)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
othArgs.add_argument("-qcplot",help="Plot focused and defocuse probablilities (y or [n])",type=str)
othArgs.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
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
fltdat = load_all_unlabeled_data(args.lblptch)

# Get data shape
nex = focdat.shape[0]; nzp = focdat.shape[1]; nxp = focdat.shape[2]
if(verb): print("Total number of examples in a file: %d"%(nex))

#for iex in range(nex):
#  fig,axarr = plt.subplots(1,3,figsize=(10,6))
#  axarr[0].imshow(focdat[iex,:,:,0],cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
#  axarr[1].imshow(defdat[iex,:,:,0],cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
#  axarr[2].imshow(fltdat[iex,:,:,0],cmap='jet',interpolation='none',vmin=0,vmax=1)
#  plt.show()

# Read in CNN architecture and weights
with open(args.arch,'r') as f:
  model = model_from_json(f.read())

wgts = model.load_weights(args.wgts)

if(verb): model.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Predict on each dataset
focprbs = model.predict(focdat,verbose=1)
defprbs = model.predict(defdat,verbose=1)

# Write the datasets to file
hff = h5py.File(args.focprb,'w')
hfd = h5py.File(args.defprb,'w')

for iex in progressbar(range(nex), "iex:"):
  datatag = create_inttag(iex,nex)
  hff.create_dataset("x"+datatag, (nxp,nzp,1), data=focprbs[iex], dtype=np.float32)
  hfd.create_dataset("x"+datatag, (nxp,nzp,1), data=defprbs[iex], dtype=np.float32)

# Close files
hff.close(); hfd.close()


