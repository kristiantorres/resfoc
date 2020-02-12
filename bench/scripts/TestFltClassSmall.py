"""
Evaluates a trained CNN that segments faults on seismic images

Inputs are the network weights and architecture as well
as the test dataset

@author: Joseph Jennings
@version: 2020.02.10
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import h5py
from deeplearn.dataloader import load_alldata
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.utils import plotseglabel, thresh
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
    "lr": 0.001,
    "nepochs": 10,
    "nflts": 32,
    "fltsize": 5,
    "unet": "y",
    "drpout": 0.0,
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

# Set defaults
parser.set_defaults(**defaults)

# Input files
ioArgs = parser.add_argument_group('Inputs and outputs')
ioArgs.add_argument("-tsdat",help="Input test dataset in H5 format",type=str)
ioArgs.add_argument("-wgtin",help="Input CNN filter coefficients",type=str)
ioArgs.add_argument("-arcin",help="Input CNN architecture",type=str)
# Other arguments
parser.add_argument("-qc",help="Plots the true label against the prediction [y]",type=str)
parser.add_argument("-nbatches",help="Number of batches on which to make predictions [default is all]",type=int)
parser.add_argument("-bsize",help="Batch size for creating data generator [20]",type=int)
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
gpus  = sep.read_list(args.gpus,[])
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)
qc = sep.yn2zoo(args.qc)

# Prediction arguments
nbatches = args.nbatches
bsize    = args.bsize

# Input output arguments
tsdat = args.tsdat

# Read in network
with open(args.arcin,'r') as f:
  model = model_from_json(f.read())

wgts = model.load_weights(args.wgtin)

if(verb): model.summary()

# Load all data
allx,ally = load_alldata(tsdat,None,bsize)
ally = np.squeeze(ally)

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Evaluate on test data (predict on all examples)
pred = model.predict(allx,verbose=1)
pred = np.squeeze(pred)
print(pred.shape)

pe = PatchExtractor((128,128),stride=(64,64))
dummy = np.zeros([512,1024])
dptch = pe.extract(dummy)

if(qc):
  nex = int(pred.shape[0]/105)
  beg = 0; end = 105
  for iex in range(nex):
    # Get the next 105 examples
    oimg = allx[beg:end,:,:]
    oimg = oimg.reshape([7,15,128,128])
    rimg = pe.reconstruct(oimg)
    plt.figure(1)
    plt.imshow(rimg,cmap='gray')
    olbl = ally[beg:end,:,:]
    olbl = olbl.reshape([7,15,128,128])
    rlbl = pe.reconstruct(olbl)
    plotseglabel(rimg,rlbl)
    oprd = pred[beg:end,:,:]
    oprd = oprd.reshape([7,15,128,128])
    rprd = pe.reconstruct(oprd)
    # Apply two sided threshold
    idxb = rprd < 0.5
    idxt = rprd > 1.5
    rprd[idxb] = 0.0; rprd[idxt] = 0.0
    idxf = rprd != 0.0
    rprd[idxf] = 1
    plotseglabel(rimg,rprd,color='blue')
    plt.show()
    beg += 105; end += 105


