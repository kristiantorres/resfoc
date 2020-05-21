"""
Trains a VGG-type network for identifying if a fault is
focused or not

@author: Joseph Jennings
@version: 2020.05.17
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import h5py
from deeplearn.dataloader import load_all_unlabeled_data, load_labeled_flat_data
from deeplearn.kerasnets import vgg3
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import random
from utils.ptyprint import progressbar
import matplotlib.pyplot as plt

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
    "gpus": [],
    "nqc": 20,
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
ioArgs.add_argument("-focdat",help="Focused data in H5 format",type=str,required=True)
ioArgs.add_argument("-defdat",help="Defocused data in H5 format",type=str,required=True)
ioArgs.add_argument("-resdat",help="Residual defocused data in H5 format",type=str,required=True)
ioArgs.add_argument("-wgtout",help="Output CNN filter coefficients",type=str)
ioArgs.add_argument("-arcout",help="Output CNN architecture",type=str)
ioArgs.add_argument("-chkpnt",help="Model checkpoint file",type=str)
ioArgs.add_argument("-lssout",help="Output loss history",type=str)
ioArgs.add_argument("-vlsout",help="Output validation loss history",type=str)
ioArgs.add_argument("-accout",help="Output accuracy history",type=str)
ioArgs.add_argument("-vacout",help="Output validation accuracy history",type=str)
# Training
trainArgs = parser.add_argument_group('Training parameters')
trainArgs.add_argument('-bsize',help='Batch size [20]',type=int)
trainArgs.add_argument('-nepochs',help='Number of passes over training data [10]',type=int)
# Other arguments
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
parser.add_argument("-nqc",help="Number of examples to QC [20]",type=int)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep()

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
gpus  = sep.read_list(args.gpus,[])
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)

# Training arguments
bsize   = args.bsize
nepochs = args.nepochs

# Load all data
focdat = load_all_unlabeled_data(args.focdat)
resdat = load_all_unlabeled_data(args.resdat)
defdat,deflbl = load_labeled_flat_data(args.defdat,None)

# Size of each dataset
nfoc = focdat.shape[0]
nres = resdat.shape[0]
ndef = defdat.shape[0]
ndiff = nfoc - ndef

# Make the three datasets the same size
idxs1 = random.sample(range(nfoc), ndiff)
idxs2 = random.sample(range(nres), ndiff)

# Delete images randomly
foctrm = np.delete(focdat,idxs1,axis=0)
restrm = np.delete(resdat,idxs2,axis=0)

# Remove half of each defocused and combine
didxs1 = random.sample(range(ndef), int(ndef/2))
didxs2 = random.sample(range(ndef), int(ndef/2))

reshlf =  np.delete(restrm,didxs1,axis=0)
defhlf =  np.delete(defdat,didxs2,axis=0)

deftot = np.concatenate([reshlf,defhlf],axis=0)

# Create labels for defocused and focused
deflbls = np.zeros(ndef)
foclbls = np.ones(ndef)

# Concatenate focused and defocused images and labels
allx = np.concatenate([deftot,foctrm],axis=0)
ally = np.concatenate([deflbls,foclbls],axis=0)

#TODO: shuffle the data first

print(allx.shape)
print(ally.shape)

ntot = allx.shape[0]

# QC the images
for iex in range(args.nqc):
  idx = np.random.randint(ntot)
  fig = plt.figure(figsize=(8,8)); ax = fig.gca()
  ax.imshow(allx[idx,:,:,0],cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5,extent=[0.0,0.64,0.64,0.0])
  ax.set_xlabel('X (km)',fontsize=16)
  ax.set_ylabel('Z (km)',fontsize=16)
  ax.tick_params(labelsize=16)
  if(ally[idx] == 0):
    print("Defocused")
    plt.savefig('./fig/defocused%d.png'%(idx),bbox_inches='tight',transparent=True,dpi=150)
  elif(ally[idx] == 1):
    print("Focused")
    plt.savefig('./fig/focused%d.png'%(idx),bbox_inches='tight',transparent=True,dpi=150)
  else:
    print("Should not be here...")
  plt.show()
  plt.close()

model = vgg3()
if(verb):
  print(model.summary())

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Create callbacks
checkpointer = ModelCheckpoint(filepath=args.chkpnt, verbose=1, save_best_only=True)

# Train the model
history = model.fit(allx,ally,epochs=nepochs,batch_size=bsize,verbose=1,shuffle=True,
                   validation_split=0.2,callbacks=[checkpointer])

# Write the model
model.save_weights(args.wgtout)

# Save the loss history
lossvepch = np.asarray(history.history['loss'])
sep.write_file(args.lssout,lossvepch)
vlssvepch = np.asarray(history.history['val_loss'])
sep.write_file(args.vlsout,vlssvepch)

# Save the accuracy history
accvepch = np.asarray(history.history['acc'])
sep.write_file(args.accout,accvepch)
vacvepch = np.asarray(history.history['val_acc'])
sep.write_file(args.vacout,vacvepch)

# Save the model architecture
with open(args.arcout,'w') as f:
  f.write(model.to_json())

