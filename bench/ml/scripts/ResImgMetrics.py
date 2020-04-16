"""
Computes Metrics for the defocused residual migration
image classification

@author: Joseph Jennings
@version: 2020.04.14
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import h5py
import numpy as np
from utils.ptyprint import create_inttag, progressbar
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.utils import plotseglabel, plotsegprobs, thresh, normalize, resample
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from resfoc.estro import estro_tgtt, refoc_tgt
from resfoc.gain import agc
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    "thresh": 0.5,
    "show": 'n',
    "gpus": [],
    "aratio": 1.0,
    "fs": 0,
    "time": "n",
    "km": "y",
    "barx": 0.91,
    "barz": 0.31,
    "hbar": 0.37,
    "xidx": 600,
    "cropsize": 154,
    "qc": "n",
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
ioArgs.add_argument("-resimgs",help="input focused images",type=str,required=True)
ioArgs.add_argument("-focimgs",help="input focused images",type=str,required=True)
ioArgs.add_argument("-fltlbls",help="Fault labels",type=str,required=True)
ioArgs.add_argument("-resptbs",help="Fault labels",type=str,required=True)
ioArgs.add_argument("-wgts",help="input CNN weights",type=str,required=True)
ioArgs.add_argument("-arch",help="input CNN architecture",type=str,required=True)
# Required arguments
ptchArgs = parser.add_argument_group('Patching parameters')
ptchArgs.add_argument('-nxo',help='Total image size',type=int,required=True)
ptchArgs.add_argument('-nzo',help='Total image size',type=int,required=True)
ptchArgs.add_argument('-nro',help='Total image size',type=int,required=True)
ptchArgs.add_argument('-oro',help='Total image size',type=float,required=True)
ptchArgs.add_argument('-dro',help='Total image size',type=float,required=True)
ptchArgs.add_argument('-ptchx',help='X dimension of patch',type=int,required=True)
ptchArgs.add_argument('-ptchz',help='Z dimension of patch',type=int,required=True)
ptchArgs.add_argument('-strdx',help='X dimension of stride',type=int,required=True)
ptchArgs.add_argument('-strdz',help='Z dimension of stride',type=int,required=True)
# Optional arguments
parser.add_argument("-qc",help="Visual QC of the predictions (y or [n])",type=str)
parser.add_argument("-thresh",help="Threshold to apply to predictions [0.5]",type=float)
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
parser.add_argument("-show",help="Show plots before saving [n]",type=str)
parser.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# SEP IO
sep = seppy.sep()

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
qc    = sep.yn2zoo(args.qc)
gpus  = sep.read_list(args.gpus,[])
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)

# Read in the network
with open(args.arch,'r') as f:
  model = model_from_json(f.read())

wgts = model.load_weights(args.wgts)

if(verb): model.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Read in the H5 datasets
hff = h5py.File(args.focimgs,'r')
hfr = h5py.File(args.resimgs,'r')
hfl = h5py.File(args.fltlbls,'r')
hfp = h5py.File(args.resptbs,'r')

# Read in each example
fkeys = list(hff.keys())
rkeys = list(hfr.keys())
lkeys = list(hfl.keys())
pkeys = list(hfp.keys())

# Number of examples
nex = len(fkeys)

# Sampling for plotting
dz = 0.01; dx = 0.01

# Create the patch extractor and the output array
nzp = args.ptchz; nxp = args.ptchx
nro = args.nro; nzo = args.nzo;   nxo = args.nxo;

per = PatchExtractor((nro,nzp,nxp),stride=(nro,args.strdz,args.strdx))
dptch = np.squeeze(per.extract(np.zeros([nro,nzo,nxo])))

pe  = PatchExtractor((nxp,nzp),stride=(args.strdx,args.strdz))
dptch = pe.extract(np.zeros([nzo,nxo]))
numpz = dptch.shape[0]; numpx = dptch.shape[1]

# Output predictions
dpreds = np.zeros([nex,numpx*numpz,nxp,nzp,1])
epreds = np.zeros([nex,numpx*numpz,nxp,nzp,1])

# Output metrics
iouds = np.zeros(nex); ioues = np.zeros(nex)

for iex in progressbar(range(nex), "iex:"):
  # Get the example
  ifoc = np.asarray(hff[fkeys[iex]]).T
  ires = np.asarray(hfr[rkeys[iex]])
  irest = np.transpose(ires,(0,2,1))
  ilbl = np.asarray(hfl[lkeys[iex]]).T
  iptb = np.asarray(hfp[pkeys[iex]]).T
  # Get the decfocused image
  idfc = ires[10,:,:].T
  # Estimate rho
  irho,oehs = estro_tgtt(irest,ifoc,args.dro,args.oro,nzp=nzp,nxp=nxp,onehot=True)
  #fig,ax = plt.subplots(1,2,figsize=(14,7))
  #ax[0].imshow(irho.T,cmap='seismic',vmin=0.97,vmax=1.03)
  #ax[1].imshow(iptb,cmap='jet',vmin=-100,vmax=100)
  #plt.show()
  irp = np.squeeze(per.extract(irest))
  iep = refoc_tgt(irp,oehs,transp=False)
  idp = pe.extract(idfc)
  #plt.figure()
  #plt.imshow(agc(iref.astype('float32')).T,cmap='gray')
  #plt.figure()
  #plt.imshow(agc(idfc.T.astype('float32')).T,cmap='gray')
  #plt.figure()
  #plt.imshow(agc(ifoc.astype('float32')).T,cmap='gray')
  #plt.show()
  # Perform the patch extraction
  ilp = pe.extract(ilbl)
  # Refocus the image
  # Reshape
  idp = idp.reshape([numpx*numpz,nzp,nxp,1])
  iep = iep.reshape([numpx*numpz,nzp,nxp,1])
  ilp = ilp.reshape([numpx*numpz,nzp,nxp,1])
  # Normalize each patch
  nidp = np.zeros(idp.shape)
  niep = np.zeros(iep.shape)
  for ip in range(numpz*numpx):
    nidp[ip,:,:] = normalize(idp[ip,:,:])
    niep[ip,:,:] = normalize(iep[ip,:,:])
  # Make a prediction
  idprd  = model.predict(nidp,verbose=0)
  ieprd  = model.predict(niep,verbose=0)
  # Save the predictions to an array
  dpreds[iex] = idprd
  epreds[iex] = ieprd
  # Threshold the predictions
  idprdt = thresh(idprd,args.thresh)
  ieprdt = thresh(ieprd,args.thresh)
  # Reconstruct the images
  idpre = pe.reconstruct(idprd.reshape([numpz,numpx,nzp,nxp]))
  iepre = pe.reconstruct(ieprd.reshape([numpz,numpx,nzp,nxp]))
  iref = pe.reconstruct(iep.reshape([numpz,numpx,nzp,nxp]))
  # Evaluate the prediction
  iouds[iex] = jaccard_score(idprdt.flatten(),ilp.flatten())
  ioues[iex] = jaccard_score(ieprdt.flatten(),ilp.flatten())
  # Plot the prediction
  if(qc):
    gdfoc = agc(idfc.astype('float32').T).T
    plotsegprobs(gdfoc,idpre,
               xlabel='X (km)',ylabel='Z (km)',xmin=0.0,xmax=(nxo)*dx,
               zmin=0,zmax=(nzo)*dz,vmin=-2.5,vmax=2.5,show=False,interp='sinc',
               pmin=0.3,alpha=0.7,ticksize=14,barlabelsize=14,barx=0.91,
               hbar=0.67,wbox=10,labelsize=14,barz=0.16)
    grfoc = agc(iref.astype('float32'))
    plotsegprobs(grfoc,iepre,
               xlabel='X (km)',ylabel='Z (km)',xmin=0.0,xmax=(nxo)*dx,
               zmin=0,zmax=(nzo)*dz,vmin=-2.5,vmax=2.5,show=True,interp='sinc',
               pmin=0.3,alpha=0.7,ticksize=14,barlabelsize=14,barx=0.91,
               hbar=0.67,wbox=10,labelsize=14,barz=0.16)

print("Mean Defocused IOU: %f"%(np.mean(iouds)))
print("Mean Resmig IOU: %f"%(np.mean(ioues)))
# Close the H5 files
hfr.close(); hfl.close(); hff.close(); hfp.close()

