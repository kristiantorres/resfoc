"""
Trains a CNN for segmenting faults on a seismic image
using the architecture by Xinming Wu

The neural network is of a U-net architecture (user can
decide for skip connections or not). Network is implemented
and trained with the Keras/Tensorflow deep learning framework.
The training is done with the ADAM optimizer within Keras

The input training data must be in H5 format. This program writes
out the model architecture as well as the model weights
and biases which can then be used subsequently for validation/testing.

@author: Joseph Jennings
@version: 2020.03.12
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import h5py
from deeplearn.dataloader import load_alldata
from deeplearn.kerasnets import unetxwu
from deeplearn.kerascallbacks import F3Pred
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "y",
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
ioArgs.add_argument("-trdat",help="Input training dataset in H5 format",type=str,required=True)
ioArgs.add_argument("-vadat",help="Input validation dataset in H5 format",type=str,required=True)
ioArgs.add_argument("-wgtout",help="Output CNN filter coefficients",type=str)
ioArgs.add_argument("-arcout",help="Output CNN architecture",type=str)
ioArgs.add_argument("-chkpnt",help="Model checkpoint file",type=str)
ioArgs.add_argument("-lssout",help="Output loss history",type=str)
ioArgs.add_argument("-vlsout",help="Output validation loss history",type=str)
ioArgs.add_argument("-accout",help="Output accuracy history",type=str)
ioArgs.add_argument("-vacout",help="Output validation accuracy history",type=str)
# Field data
f3Args = parser.add_argument_group("Paths for analyzing field data predictions")
f3Args.add_argument("-f3dat",help="Path to F3 data",type=str)
f3Args.add_argument("-f3prd",help="Path for writing predictions on F3 data",type=str)
f3Args.add_argument("-f3fig",help="Path for writing figures of predictions on F3 data",type=str)
# Training
trainArgs = parser.add_argument_group('Training parameters')
trainArgs.add_argument('-bsize',help='Batch size [20]',type=int)
trainArgs.add_argument('-nepochs',help='Number of passes over training data [10]',type=int)
# Other arguments
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
gpus  = sep.read_list(args.gpus,[])
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)

# Training arguments
bsize   = args.bsize
nepochs = args.nepochs

# Load all data
allx,ally = load_alldata(args.trdat,None,32)
xshape = allx.shape[1:]
yshape = ally.shape[1:]

model = unetxwu()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Create callbacks
checkpointer = ModelCheckpoint(filepath=args.chkpnt, verbose=1, save_best_only=True)
f3predicter  = F3Pred(args.f3dat,args.f3prd,args.f3fig,105,(128,128),(64,64))

# Train the model
history = model.fit(allx,ally,epochs=nepochs,batch_size=bsize,verbose=1,shuffle=True,
                   validation_split=0.2,callbacks=[checkpointer,f3predicter])

# Write the model
model.save_weights(args.wgtout)

# Save the loss history
lossvepch = np.asarray(history.history['loss'])
laxes = seppy.axes([len(lossvepch)],[0.0],[1.0])
sep.write_file(None,laxes,lossvepch,args.lssout)
vlssvepch = np.asarray(history.history['val_loss'])
sep.write_file(None,laxes,vlssvepch,args.vlsout)

# Save the accuracy history
accvepch = np.asarray(history.history['acc'])
aaxes = seppy.axes([len(accvepch)],[0.0],[1.0])
sep.write_file(None,aaxes,accvepch,args.accout)
vacvepch = np.asarray(history.history['val_acc'])
sep.write_file(None,aaxes,vacvepch,args.vacout)

# Save the model architecture
with open(args.arcout,'w') as f:
  f.write(model.to_json())

