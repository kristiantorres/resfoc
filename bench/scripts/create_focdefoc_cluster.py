"""
Creates focused and defocused images

@author: Joseph Jennings
@version: 2020.05.11
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from cluster.focdefocslrm import focdefocjob
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
    "outdir": "/data/sep/joseph29/projects/resfoc/bench/dat/focdefoc",
    "prefix": "refoc",
    "dpath": "/data/sep/joseph29/scratch/focdefoc",
    "fx": 120,
    "nxw": 1024,
    "na": 64,
    "tjobs": 1000,
    "ajobs": 200,
    "nprocs": 16,
    "nsubmit": 5,
    "logpath": "./log/focdefoc",
    "parpath": "./par/focdefoc",
    "jobprefix": "focdefoc",
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
ioArgs = parser.add_argument_group('Output files')
ioArgs.add_argument("-outdir",help="Output directory where to write the files",type=str,required=True)
ioArgs.add_argument("-prefix",help="Prefix for output file names",type=str,required=True)
ioArgs.add_argument("-dpath",help="Output datapath of where to write the SEPlib binaries",type=str,required=True)
ioArgs.add_argument("-beg",help="Numeric suffix for keeping track of examples [0]",type=int,required=True)
# Window arguments
winArgs = parser.add_argument_group('Windowing parameters')
winArgs.add_argument("-fx",help="First x sample [0]",type=int)
winArgs.add_argument("-nxw",help="Length of window in x [1300]",type=int)
winArgs.add_argument("-na",help="Number of angles to compute [64]",type=int)
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
maxnum = 9999

# Base command for all jobs
bcmd = '/data/biondo/joseph29/opt/anaconda3/envs/py35/bin/python ./scripts/randund_fault_focus.py -c '

# Create and submit all jobs
sepqfull = False; hr2qfull= False
actjobs = []; lefjobs = []
# Starting queue
squeue = 'sep'
for ijob in range(tjobs):
  if(ijob < ajobs):
    # Create job
    actjobs.append(focdefocjob(args,jobprefix,maxnum,parpath,logpath,verb=verb))
    args.beg += 1
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
    args.beg += 1
    lefjobs.append(focdefocjob(args,jobprefix,maxnum,parpath,logpath,verb=verb))

if(verb): print("%d jobs submitted, %d jobs waiting. Managing jobs now...\n"%(len(actjobs),len(lefjobs)))

# Send jobs to job manager
manage_slurmjobs(actjobs,bcmd,jobprefix,lefjobs,ajobs,nprocs,nleft,args.nsubmit,sepqfull,hr2qfull,delay=delay,verb=verb)

# Clean cluster files
if(klean):
  clean_clusterfiles(jobprefix,logpath,parpath)

