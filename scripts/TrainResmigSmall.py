"""
Trains a CNN for reconstructing a spatially-varying focusing
parameter (rho) from residually migrated images  that have
been migrated with different values for rho.

The neural network is of a U-net architecture (user can
decide for skip connections or not). Network is implemented
and trained with the Keras/Tensorflow deep learning framework.
The training is done with the ADAM optimizer within Keras

The input training data must be in H5 format. This program writes
out the model architecture as well as the model weights
and biases which can then be used subsequently for validation/testing.

The difference between this program and the program TrainResmig.py
is that this one assumes the data can all be loaded into RAM
(no data generator is needed).

@author: Joseph Jennings
@version: 2020.01.05
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import h5py
from deeplearn.dataloader import load_alldata
from deeplearn.kerasnets import auto_encoder
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

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
ioArgs.add_argument("-trdat",help="Input training dataset in H5 format",type=str)
ioArgs.add_argument("-vadat",help="Input validation dataset in H5 format",type=str)
ioArgs.add_argument("-wgtout",help="Output CNN filter coefficients",type=str)
ioArgs.add_argument("-arcout",help="Output CNN architecture",type=str)
trainArgs = parser.add_argument_group('Training parameters')
trainArgs.add_argument('-lr',help='Learning rate [0.001]',type=float)
trainArgs.add_argument('-bsize',help='Batch size [20]',type=int)
trainArgs.add_argument('-nepochs',help='Number of passes over training data [10]',type=int)
netargs = parser.add_argument_group('CNN design parameters')
netargs.add_argument('-nflts',help='Number of filters that will be created after first conv [32]',type=int)
netargs.add_argument('-fltsize', help='Size of square convolutional filter [5]',type=int)
netargs.add_argument('-unet',help='Create a network with skip connects [y]',type=str)
netargs.add_argument('-drpout',help='Dropout percent from 0 - 1 [0.0]',type=float)
# Other arguments
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb  = sep.yn2zoo(args.verb)

# Training arguments
lr      = args.lr
bsize   = args.bsize
nepochs = args.nepochs

# Network arguments
nf    = args.nflts
fsize = args.fltsize
unet  = sep.yn2zoo(args.unet)
drpot = args.drpout

# Load all data
allx,ally = load_alldata(args.trdat,args.vadat,bsize)
xshape = allx.shape[1:]
yshape = ally.shape[1:]

# Create the keras model
model = auto_encoder(yshape, xshape[2], 1, nf, fsize, unet, drpot)
if(verb): model.summary()

# Create the optimization object
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile the model
model.compile( loss='mean_squared_error', optimizer=opt )

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Train the model
history = model.fit(allx,ally,epochs=nepochs,batch_size=bsize,verbose=1,shuffle=True,validation_split=0.2)

# Write the model
model.save_weights(args.wgtout)

# Save the model architecture
with open(args.arcout,'w') as f:
  f.write(model.to_json())

