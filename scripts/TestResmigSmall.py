"""
Evaluates a trained CNN that estimates rho (focusing parameter)
from residually migrated images.

Inputs are the network weights and architecture as well
as the test dataset. The network is then evaluated on the
test dataset and outputs the estimated rho. These rho
values should then be used to remigrate the dataset
to assess the quality of the image.

The input test dataset is expected to be of
H5 format. The neural network weights
should also be in H5 format and the architecture
should be a JSON file. The rhos can either be output
in H5 or SEPlib format (default is SEPlib format).

This version loads all of the test data into memory

@author: Joseph Jennings
@version: 2020.01.05
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import h5py
from deeplearn.dataloader import resmig_generator_h5
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "y",
    "nbatches": None,
    "bsize": 20,
    "seplib": "y",
    "qc": "y",
    "roout": None,
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
ioArgs.add_argument("-tsdat",help="Input test dataset in H5 format",type=str)
ioArgs.add_argument("-wgtin",help="Output CNN filter coefficients",type=str)
ioArgs.add_argument("-arcin",help="Output CNN architecture",type=str)
ioArgs.add_argument("-roout",help="Output estimated rho",type=str)
# Other arguments
parser.add_argument("-qc",help="Plots the estimated rho against the true rho",type=str)
parser.add_argument("-nbatches",help="Number of batches on which to make predictions [default is all]",type=int)
parser.add_argument("-bsize",help="Batch size for creating data generator [20]",type=int)
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-seplib",help="Flag whether to output data in [SEPlib] or H5 format",type=str)
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb = sep.yn2zoo(args.verb)
sepf = sep.yn2zoo(args.seplib)
qc   = sep.yn2zoo(args.qc)

# Prediction arguments
nbatches = args.nbatches
bsize    = args.bsize

# Input output arguments
tsdat = args.tsdat
roout = args.roout

# Read in network
with open(args.arcin,'r') as f:
  model = model_from_json(f.read())

wgts = model.load_weights(args.wgtin)

if(verb): model.summary()

# Create dataloader
tsdgen = resmig_generator_h5(tsdat,bsize)
xshape,yshape = tsdgen.get_xyshapes()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Evaluate on test data (predict on all examples)
pred = model.predict_generator(tsdgen,verbose=1,steps=nbatches)
pred = np.squeeze(pred)

# Write the predictions
if(roout != None):
  if(sepf):
    pred = np.transpose(pred,(1,2,0))
    # First create the axes
    nz = pred.shape[0]; nx = pred.shape[1]; nex = pred.shape[2]
    axes = seppy.axes([nz,nx,nex],[0.0,0.0,0.0],[1.0,1.0,1.0])
    # Write the data
    sep.write_file(None,axes,pred,ofname=roout,dpath='/net/fantastic/scr2/joseph29/')
  else:
    with h5py.File(roout,'w') as hf:
      nex = pred.shape[0]; nz = pred.shape[1]; nx = pred.shape[2]
      hf.create_dataset('pred', (nex,nz,nx), data=pred, dtype=np.float32)

if(qc):
  with h5py.File(tsdat,'r') as hf:
    keys = list(hf.keys());
    nb = int(len(keys)/2)
    nex = pred.shape[0]; k = 0
    if(nbatches == None):
      nbatches = nb
    for ib in range(nbatches):
      for iex in range(bsize):
        f,ax = plt.subplots(1,2,figsize=(10,5),gridspec_kw={'width_ratios': [1, 1]})
        ax[0].imshow(pred[k,:,:],cmap='jet',vmin=0.95,vmax=1.05)
        ax[1].imshow(hf[keys[ib+nb]][iex],cmap='jet',vmin=0.95,vmax=1.05)
        plt.savefig('./fig/ex%d.png'%(iex),bbox_inches='tight',dpi=150)
        plt.show()
        k += 1
