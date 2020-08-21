"""
Labels angle gather patches as defocused or focused

@author: Joseph Jennings
@version: 2020.05.25
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import h5py
from genutils.ptyprint import progressbar, create_inttag
import numpy as np
from deeplearn.utils import plotseglabel, plotsegprobs, normalize
from deeplearn.dataloader import load_all_unlabeled_data,load_unlabeled_flat_data
from deeplearn.focuslabels import corrsim, semblance_power
from genutils.plot import plot_cubeiso
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "focptch": "",
    "defptch": "",
    "focprb": "" ,
    "defprb": "",
    "begex": 0,
    "endex": -1,
    "nqc": 100,
    "thresh1": 0.7,
    "thresh2": 0.5,
    "thresh3": 0.7,
    "pixthresh": 20,
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
ioArgs.add_argument("-fltptch",help="H5 file containing fault label patches",type=str,required=True)
ioArgs.add_argument("-focprb",help="H5 file containing focused fault probabilities",type=str,required=True)
ioArgs.add_argument("-defprb",help="H5 file containing defocused fault probabilities",type=str,required=True)
ioArgs.add_argument("-begex",help="Beginning example to process [0]",type=int)
ioArgs.add_argument("-endex",help="Ending example to process [last example]",type=int)
ioArgs.add_argument("-deflbls",help="Output H5 file containing labeled defocused patches",type=str,required=True)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-pixthresh",help="Number of pixels in a patch to determine if it has a fault [20]",type=int)
othArgs.add_argument("-thresh1",help="Image correlation threshold for determining if image is defocused [0.7]",type=float)
othArgs.add_argument("-thresh2",help="Fault probability threshold for determining if image is defocused [0.5]",type=float)
othArgs.add_argument("-thresh3",help="Angle stack threshold for determining if image is defocused [0.7]",type=float)
othArgs.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
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

# Beginning and ending examples
begex = args.begex; endex = args.endex

# Read in patch data
focdat = load_all_unlabeled_data(args.focptch,begex,endex)
defdat = load_all_unlabeled_data(args.defptch,begex,endex)
fltdat = load_all_unlabeled_data(args.fltptch,begex,endex)

# Get data shape
nex = focdat.shape[0]; nang = focdat.shape[1]; nzp = focdat.shape[2]; nxp = focdat.shape[3]
if(verb): print("Total number of examples in a file: %d"%(nex))

# Determine number of examples in a single image
nimg = nex/(endex-begex)

# Stack over angles
focstk = np.sum(focdat,axis=1)
defstk = np.sum(defdat,axis=1)

# Load in fault probabilities
pbegex = int(nimg*begex); pendex = int(nimg*endex)
focprb = load_unlabeled_flat_data(args.focprb,pbegex,pendex)
defprb = load_unlabeled_flat_data(args.defprb,pbegex,pendex)

os = [-70.0,0.0,0.0]; ds = [2.22,0.01,0.01]
# QC the predictions
for iex in progressbar(range(args.nqc),"nqc:"):
  idx = np.random.randint(nex)
  # Compute metrics
  corrimg = corrsim(defstk[idx,:,:,0],focstk[idx,:,:,0])
  corrprb = corrsim(defprb[idx,:,:,0],focprb[idx,:,:,0])
  fsemb = semblance_power(focdat[iex,:,:,:,0])
  dsemb = semblance_power(defdat[iex,:,:,:,0])
  sembrat = dsemb/fsemb
  # Plot images
  plotsegprobs(focstk[idx,:,:,0],focprb[idx,:,:,0],interp='sinc',show=False,pmin=0.2,title='CORRIMG=%.2f'%(corrimg))
  plotsegprobs(defstk[idx,:,:,0],defprb[idx,:,:,0],interp='sinc',show=False,pmin=0.2,title='CORRPRB=%.2f'%(corrprb))
  plotseglabel(focstk[idx,:,:,0],fltdat[idx,:,:,0],interp='sinc',show=False)
  plot_cubeiso(focdat[idx,:,:,:,0],os=os,ds=ds,stack=True,elev=15,show=False,verb=False)
  plot_cubeiso(defdat[idx,:,:,:,0],os=os,ds=ds,stack=True,elev=15,show=False,verb=False,title='SEMBPWR=%.2f'%(sembrat))
  plt.show()

# Output list of defocused images
defout = []

print("Starting at %d and ending at %d"%(begex,endex))
for iex in progressbar(range(nex), "nsim:"):
  # Compute fault metrics
  fltnum = np.sum(fltdat[iex,:,:,0])
  # If example has faults, use fault criteria
  if(fltnum > args.pixthresh):
    corrimg = corrsim(defstk[iex,:,:,0],focstk[iex,:,:,0])
    corrprb = corrsim(defprb[iex,:,:,0],focprb[iex,:,:,0])
    if(corrimg < args.thresh1 and corrprb < args.thresh1):
      defout.append(defdat[iex,:,:,:,0])
    elif(corrimg < args.thresh2 or corrprb < args.thresh2):
      defout.append(defdat[iex,:,:,:,0])
  else:
    # Compute angle metrics
    fsemb = semblance_power(focdat[iex,:,:,:,0])
    dsemb = semblance_power(defdat[iex,:,:,:,0])
    sembrat = dsemb/fsemb
    if(sembrat < args.thresh3):
      defout.append(defdat[iex,:,:,:,0])

print("Keeping def=%d examples"%(len(defout)))

# Convert to numpy array
defs = np.asarray(defout)

ntot = defs.shape[0]

# Write the labeled data to file
hfd = h5py.File(args.deflbls,'w')

for iex in progressbar(range(ntot), "iex:"):
  datatag = create_inttag(iex,nex)
  hfd.create_dataset("x"+datatag, (nang,nxp,nzp,1), data=np.expand_dims(defs[iex],axis=-1), dtype=np.float32)
  hfd.create_dataset("y"+datatag, (1,), data=0, dtype=np.float32)

# Close file
hfd.close()

print("Success!")

