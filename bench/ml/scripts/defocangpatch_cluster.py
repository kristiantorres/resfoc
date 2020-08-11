"""
Labels defocused angle gather fault image patches

@author: Joseph Jennings
@version: 2020.05.26
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from cluster.defocanglblslrm import defocanglbljob
import cluster.slurmhelper as slurm
from cluster.slurmmanager import manage_slurmjobs, clean_clusterfiles
import time
import subprocess, glob

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "focptch": "/data/sep/joseph29/projects/resfoc/bench/dat/angfaultfocptch.h5",
    "defptch": "/data/sep/joseph29/projects/resfoc/bench/dat/angfaultdefptch.h5",
    "focprb": "/data/sep/joseph29/projects/resfoc/bench/dat/stkfocfltprb.h5",
    "defprb": "/data/sep/joseph29/projects/resfoc/bench/dat/stkdeffltprb.h5",
    "oprefix": "defocanglbl",
    "thresh1": 0.7,
    "thresh2": 0.5,
    "thresh3": 0.7,
    "tjobs": 1000,
    "ajobs": 200,
    "nprocs": 16,
    "nsubmit": 5,
    "logpath": "./log/defocanglbl",
    "parpath": "./par/defocanglbl",
    "jobprefix": "defocanglbl",
    "nleft": 2,
    "delay": 10.0,
    "klean": 'y',
    "blacklist": [''],
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
ioArgs = parser.add_argument_group('Input output')
ioArgs.add_argument("-focptch",help="Input focused image H5 file",type=str,required=True)
ioArgs.add_argument("-defptch",help="Input defocused image H5 file",type=str,required=True)
ioArgs.add_argument("-focprb",help="Input focused image fault probablility H5 file",type=str,required=True)
ioArgs.add_argument("-defprb",help="Input defocused image fault probability H5 file",type=str,required=True)
ioArgs.add_argument("-oprefix",help="Prefix for output label H5 file",type=str,required=True)
ioArgs.add_argument("-begex",help="First example for processing [0]",type=int,required=True)
ioArgs.add_argument("-nex",help="Number of examples per job [10500]",type=int,required=True)
ioArgs.add_argument("-totex",help="Total number of examples [104790]",type=int,required=True)
# Labeling arguments
lblArgs = parser.add_argument_group('Labeling parameters')
lblArgs.add_argument("-pixthresh",help="Number of pixels to determine if patch has a fault [20]",type=int)
lblArgs.add_argument("-thresh1",help="First threshold [0.7]",type=float)
lblArgs.add_argument("-thresh2",help="Second threshold [0.5]",type=float)
lblArgs.add_argument("-thresh3",help="Third threshold [0.7]",type=float)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
# Cluster arguments
cluArgs = parser.add_argument_group('Cluster parameters')
cluArgs.add_argument("-tjobs",help="Total number of jobs to run [600]",type=int)
cluArgs.add_argument("-ajobs",help="Number of jobs either in queue or running at once [200]",type=int)
cluArgs.add_argument("-nprocs",help="Number of processors to use per node [8]",type=int)
cluArgs.add_argument("-nsubmit",help="Number of times to attempt a job submission [5]",type=int)
cluArgs.add_argument("-logpath",help="Path to logfile [current directory]",type=str)
cluArgs.add_argument("-parpath",help="Path to parfile [current directory]",type=str)
cluArgs.add_argument("-jobprefix",help="Job prefix for par files [refoc]",type=str)
cluArgs.add_argument("-nleft",help="Number of jobs to be queued in each queue [2]",type=int)
cluArgs.add_argument("-delay",help="Amount of time in seconds to wait between prints [10]",type=float)
cluArgs.add_argument("-klean",help="Clean up cluster submission files [y]",type=str)
cluArgs.add_argument("-blacklist",help="Nodes that the user does not want to use",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Setup IO
sep = seppy.sep()

# Get command line arguments
tjobs = args.tjobs; ajobs = args.ajobs
nprocs = args.nprocs
logpath = args.logpath; parpath = args.parpath
jobprefix = args.jobprefix
verb = sep.yn2zoo(args.verb);
klean = sep.yn2zoo(args.klean)
nleft = args.nleft
delay = args.delay
blacklist = sep.read_list(args.blacklist,default=[''],dtype='str')
# Block node maz032
maxnum = 9999

# Base command for all jobs
bcmd = '/data/biondo/joseph29/opt/anaconda3/envs/py35/bin/python ./scripts/defocpatchang_data.py -c '

# Create and submit all jobs
sepqfull = False; hr2qfull= False
actjobs = []; lefjobs = []
# Starting queue
squeue = 'sep'
for ijob in range(tjobs):
  if(ijob < ajobs):
    # Create job
    actjobs.append(defocanglbljob(args,jobprefix,maxnum,parpath,logpath,verb=verb))
    args.begex += args.nex
    # Submit job
    cmd = bcmd + actjobs[ijob].pfname
    actjobs[ijob].submit(jobprefix,cmd,nprocs=nprocs,queue=squeue,sleep=2)
    # Get the status of the queues
    qlines = slurm.qstat()
    sepq = slurm.get_numjobs('sep',qfile=qlines)
    hr2q = slurm.get_numjobs('twohour',qfile=qlines)
    if(verb):
      print("sep queue: %d R %d Q %d C"%(sepq['R'],sepq['Q'],sepq['C']))
      print("2hr queue: %d R %d Q %d C"%(hr2q['R'],hr2q['Q'],hr2q['C']))
    if(sepq['Q'] >= 2):
      squeue = 'twohour'
      sepqfull = True
    if(hr2q['Q'] >= 2):
      ajobs = sepq['R'] + hr2q['R']
      hr2qfull = True
    if(verb): print("Job=%d %s"%(ijob, actjobs[ijob].jobid))
  else:
    # Leftover jobs, to be submitted
    args.begex += args.nex
    lefjobs.append(defocanglbljob(args,jobprefix,maxnum,parpath,logpath,verb=verb))

if(verb): print("%d jobs submitted, %d jobs waiting. Managing jobs now...\n"%(len(actjobs),len(lefjobs)))

# Send jobs to job manager
manage_slurmjobs(actjobs,bcmd,jobprefix,lefjobs,ajobs,    # Jobs
                 nprocs,nleft,args.nsubmit,               # Job parameters
                 sepqfull,hr2qfull,delay=delay,verb=verb) # Queues and other

# Clean cluster files
if(klean):
  clean_clusterfiles(jobprefix,logpath,parpath)

