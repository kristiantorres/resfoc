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
@version: 2020.01.06
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import h5py
from deeplearn.dataloader import load_alldata
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
    "gpus": [],
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
parser.add_argument("-qc",help="Plots the estimated rho against the true rho [y]",type=str)
parser.add_argument("-nbatches",help="Number of batches on which to make predictions [default is all]",type=int)
parser.add_argument("-bsize",help="Batch size for creating data generator [20]",type=int)
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-seplib",help="Flag whether to output data in [SEPlib] or H5 format",type=str)
parser.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb = sep.yn2zoo(args.verb)
gpus = sep.read_list(args.gpus,[])
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)
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

# Load all data
allx,ally = load_alldata(tsdat,None,bsize)
ally = np.squeeze(ally)

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Evaluate on test data (predict on all examples)
pred = model.predict(allx,verbose=1)
pred = np.squeeze(pred)
print(pred.shape)

# Write the predictions
if(roout != None):
  if(sepf):
    predt = np.transpose(pred,(1,2,0))
    # First create the axes
    nz = predt.shape[0]; nx = predt.shape[1]; nex = predt.shape[2]
    axes = seppy.axes([nz,nx,nex],[0.0,0.0,0.0],[1.0,1.0,1.0])
    # Write the data
    sep.write_file(None,axes,predt,ofname=roout,dpath='/net/fantastic/scr2/joseph29/')
  else:
    with h5py.File(roout,'w') as hf:
      nex = pred.shape[0]; nz = pred.shape[1]; nx = pred.shape[2]
      hf.create_dataset('pred', (nex,nz,nx), data=pred, dtype=np.float32)

#TODO: remigrate with the new prediction
if(qc):
  with h5py.File(tsdat,'r') as hf:
    keys = list(hf.keys());
    nb = int(len(keys)/2)
    nex = pred.shape[0]; k = 0
    if(nbatches == None):
      nbatches = nb
    for ib in range(nbatches):
      for iex in range(bsize):
        f,ax = plt.subplots(1,3,figsize=(15,8),gridspec_kw={'width_ratios': [1, 1, 1]})
        im1 = ax[0].imshow(allx[k,:,:,5],cmap='gray',extent=[0,255*20/1000,127*20/1000,0.0])
        ax[0].set_xlabel('X (km)',fontsize=18)
        ax[0].set_ylabel('Z (km)',fontsize=18)
        ax[0].set_title(r'$\rho$=1',fontsize=18)
        ax[0].tick_params(labelsize=18)
        im2 = ax[1].imshow(ally[k,:,:],cmap='jet',vmin=0.95,vmax=1.05,extent=[0,255*20/1000,127*20/1000,0.0])
        ax[1].set_xlabel('X (km)',fontsize=18)
        ax[1].set_title('Label',fontsize=18)
        ax[1].tick_params(labelsize=18)
        im3 = ax[2].imshow(pred[k,:,:],cmap='jet',vmin=0.95,vmax=1.05,extent=[0,255*20/1000,127*20/1000,0.0])
        ax[2].set_xlabel('X (km)',fontsize=18)
        ax[2].set_title('Prediction',fontsize=18)
        ax[2].tick_params(labelsize=18)
        cbar_ax = f.add_axes([0.91,0.39,0.02,0.21])
        cbar = f.colorbar(im3,cbar_ax,format='%.2f')
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(r'$\rho$',fontsize=18)
        cbar.draw_all()
        plt.savefig('./fig/ex50-%d.png'%(k),bbox_inches='tight',dpi=150)
        plt.show()
        k += 1

