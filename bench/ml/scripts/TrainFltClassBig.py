"""
Trains a CNN for segmenting faults on a seismic image

The neural network is of a U-net architecture (user can
decide for skip connections or not). Network is implemented
and trained with the Keras/Tensorflow deep learning framework.
The training is done with the ADAM optimizer within Keras

The input training data must be in H5 format. This program writes
out the model architecture as well as the model weights
and biases which can then be used subsequently for validation/testing.

@author: Joseph Jennings
@version: 2020.03.02
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import h5py
from deeplearn.dataloader import load_alldata
from deeplearn.kerasnets import auto_encoder
from deeplearn.keraslosses import wgtbce
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
    "lr": 0.001,
    "nepochs": 10,
    "classwgt1": 0.5,
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
ioArgs.add_argument("-lssout",help="Output loss history",type=str)
ioArgs.add_argument("-vlsout",help="Output validation loss history",type=str)
ioArgs.add_argument("-accout",help="Output accuracy history",type=str)
ioArgs.add_argument("-vacout",help="Output validation accuracy history",type=str)
# Training
trainArgs = parser.add_argument_group('Training parameters')
trainArgs.add_argument('-lr',help='Learning rate [0.001]',type=float)
trainArgs.add_argument('-bsize',help='Batch size [20]',type=int)
trainArgs.add_argument('-nepochs',help='Number of passes over training data [10]',type=int)
trainArgs.add_argument('-classwgt1',help='Class weight to balance cross entropy loss [0.5]',type=float)
# Network
netargs = parser.add_argument_group('CNN design parameters')
netargs.add_argument('-nflts',help='Number of filters that will be created after first conv [32]',type=int)
netargs.add_argument('-fltsize', help='Size of square convolutional filter [5]',type=int)
netargs.add_argument('-unet',help='Create a network with skip connects [y]',type=str)
netargs.add_argument('-drpout',help='Dropout percent from 0 - 1 [0.0]',type=float)
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
lr      = args.lr
bsize   = args.bsize
nepochs = args.nepochs

# Network arguments
nf    = args.nflts
fsize = args.fltsize
unet  = sep.yn2zoo(args.unet)
drpot = args.drpout

# Load all data
allx,ally = load_alldata(args.trdat,None,32)
xshape = allx.shape[1:]
yshape = ally.shape[1:]

# Create the keras model
model = auto_encoder(yshape, xshape[2], 1, nf, fsize, unet, drpot, False)
if(verb): model.summary()

# Create the optimization object
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile the model
#model.compile( loss='binary_crossentropy', optimizer=opt )
model.compile( loss=wgtbce(args.classwgt1), optimizer=opt , metrics=['accuracy'] )

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Train the model
history = model.fit(allx,ally,epochs=nepochs,batch_size=bsize,verbose=1,shuffle=True,
                   validation_split=0.2)

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

