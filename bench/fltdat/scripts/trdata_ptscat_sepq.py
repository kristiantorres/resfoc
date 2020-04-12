"""
Creates point scatterer training data for deep learning residual migration.
The idea is to replicate (with deep learning) the experiments performed in the paper:
Velocity estimation by image-focusing analysis, Biondi 2010.

Outputs the labels and the features as separate SEPlib files. If a large number of
examples are desired, the program will write a file after every 50 examples have
been obtained. The output of each label file is the rho field as a function of
x and z. The features output are residually migrated prestack images.

For imaging, a split spread acquisition is always assumed where receivers are placed
at each gridpoint and sources are placed at every 10 gridpoints. Default sampling
intervals are dx=20m and dz=20m

The wavelet used for modeling is a ricker with central frequency of 15Hz (max around 30Hz)

The migration velocity is always constant (v=2500 m/s) and the modeling
velocity varies around 2500 m/s determined by the rho axis chosen (nro,oro,dro)

For the residual migration, the nro provided is just one side of the
residual migration. So the actual output of residual migration parameters is
2*nro-1 (again to enforce symmetry) and therefore the actual oro is computed
as: oro - (nro-1)*dro. This forces that the output be centered at the oro
provided by the user

Finally, this script is designed for submitting many jobs on a PBS/Torque
cluster. The user must provide the total number of active nodes desired
at a given time and the queue that is desired. It is recommended that
different scripts be run for different queues.

@author: Joseph Jennings
@version: 2019.12.27
"""

import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from resfoc.training_data import createdata_ptscat
import clusterhelp as pbs
import h5py

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "nex": 50,
    "nwrite": 50,
    "nsx": 26,
    "osx": 3,
    "nz": 256,
    "nx": 256,
    "nh": 10,
    "nro": 6,
    "oro": 1.0,
    "dro": 0.01,
    "verb": 'y',
    "nprint": 100,
    "prefix": "",
    "beg": 0,
    "end": 9999,
    "njobs": 30,
    "tjobs": 145,
    "queue": 'sep',
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
ioArgs = parser.add_argument_group('Output files')
ioArgs.add_argument("-outdir",help="Output directory of where to write the SEPlib files",type=str)
ioArgs.add_argument("-datapath",help="Output datapath of where to write the SEPlib binaries",type=str)
ioArgs.add_argument("-nwrite",help="Number of examples to compute before writing [20]",type=int)
ioArgs.add_argument("-prefix",help="Prefix that will be used for label and feature files [None]",type=str)
ioArgs.add_argument("-beg",help="Numeric suffix used for keeping track of examples [0]",type=int)
ioArgs.add_argument("-end",help="Last example for writing [9999]",type=int)
# Imaging parameters
imgArgs = parser.add_argument_group('Imaging parameters')
imgArgs.add_argument("-nsx",help="Number of sources [41]",type=int)
imgArgs.add_argument("-osx",help="First source point [0 samples]",type=int)
imgArgs.add_argument("-nz",help="Number of depth samples of image [256]",type=int)
imgArgs.add_argument("-nx",help="Number of lateral samples of image [500]",type=int)
imgArgs.add_argument("-nh",help="Number of subsurface offsets of image [10]",type=int)
# Residual migration parameters
rmigArgs = parser.add_argument_group('Residual migration parameters')
rmigArgs.add_argument("-nro",help="Number of residual migrations [6]",type=int)
rmigArgs.add_argument("-oro",help="Center residual migration [1.0]",type=float)
rmigArgs.add_argument("-dro",help="Rho spacing [0.01]",type=float)
# Machine learning parameters
mlArgs = parser.add_argument_group('Machine learning parameters')
mlArgs.add_argument("-nex",help="Total number of examples [30]",type=int)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-verb",help="Verbosity flag [y]",type=str)
othArgs.add_argument("-nprint",help="How often to print a new example [100]",type=int)
othArgs.add_argument("-tjobs",help="Total number of jobs to run [145]",type=int)
othArgs.add_argument("-njobs",help="Number of jobs to keep active at once (one job per node) [30]",type=int)
othArgs.add_argument("-queue",help="Queue on which to submit job [sep]",type=str)
othArgs.add_argument("-trial",help="Run a trial (no job submissions) for testing",type=str)
args = parser.parse_args(remaining_argv)

# Setup IO
sep = seppy.sep(sys.argv)

# One example takes about 3 min and is approximately 100MB
# On default queue, max time is 2hr. => One job can run 40 examples max
# 40*100MB/1024 =~ 4GB which is OK for keeping in memor
# To stay safe, 35 examples is probably good for one job

# Need 10000 examples in total => approximately 290 jobs
# Split this over sep and general queues where SEP has 30
# jobs and lets say general also has 30. So we can split each in two
# about 145 total jobs per queue

# Get the command line arguments
tjobs = args.tjobs; njobs = args.njobs
queue = args.queue
trial = sep.yn2zoo(args.trial)
verb = sep.yn2zoo(args.verb)

# Current number of jobs
totjobs = 0
k = 0
myjobs = pbs.get_numjobs('joseph29',queue)

bcmd = '/data/sep/joseph29/opt/anaconda3/envs/py35/bin/python ./scripts/trdata_ptscat_allsep.py -c '

while(totjobs < tjobs):
  while(pbs.get_numjobs('joseph29',queue) <= njobs):
    # Write the par file
    parname = './par/ptscat%d.par'%(k)
    pbs.write_trdatpar(parname,args)
    # Write the PBS script
    cmd = bcmd + parname
    sfile = pbs.write_script('ptscat%d'%(k),cmd,queue=queue)
    # Submit the job
    pbs.submit_job(sfile,submit=trial,verb=verb)
    # Increase counters
    k += nex; totjobs += 1

