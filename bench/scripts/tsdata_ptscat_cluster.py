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

Finally, this script provides the capability of submitting many jobs
to a PBS/Torque cluster. By default, the script attempts run 200 jobs
(total examples = no. of jobs * nexamples per job). The user can
override these parameters in the cluster arguments section

@author: Joseph Jennings
@version: 2019.12.31
"""

import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import cluster.rmtrdat as rmtrdat
import cluster.pbshelper as pbs
import subprocess

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "nex": 15,
    "nwrite": 15,
    "nsx": 41,
    "osx": 3,
    "nz": 256,
    "nx": 400,
    "nh": 10,
    "nro": 6,
    "oro": 1.0,
    "dro": 0.01,
    "verb": 'y',
    "nprint": 100,
    "prefix": "",
    "beg": 0,
    "end": 9999,
    "tjobs": 600,
    "ajobs": 200,
    "nsubmit": 5,
    "logpath": ".",
    "parpath": ".",
    "jobprefix": "ptscat0",
    "klean": 'y'
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
mlArgs.add_argument("-nex",help="Total number of examples [50]",type=int)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-verb",help="Verbosity flag [y]",type=str)
othArgs.add_argument("-nprint",help="How often to print a new example [100]",type=int)
# Cluster arguments
cluArgs = parser.add_argument_group('Cluster parameters')
cluArgs.add_argument("-tjobs",help="Total number of jobs to run [600]",type=int)
cluArgs.add_argument("-ajobs",help="Number of jobs either in queue or running at once [200], cannot do more than 200 in default queue",type=int)
cluArgs.add_argument("-nsubmit",help="Number of times to attempt a job submission [5]",type=int)
cluArgs.add_argument("-logpath",help="Path to logfile [current directory]",type=str)
cluArgs.add_argument("-parpath",help="Path to parfile [current directory]",type=str)
cluArgs.add_argument("-jobprefix",help="Job prefix for par files [ptscat0]",type=str)
cluArgs.add_argument("-klean",help="Clean up cluster submission files [y]",type=str)
args = parser.parse_args(remaining_argv)

# Setup IO
sep = seppy.sep(sys.argv)

# Get command line arguments
tjobs = args.tjobs; ajobs = args.ajobs
nex = args.nex
logpath = args.logpath; parpath = args.parpath
jobprefix = args.jobprefix
verb = sep.yn2zoo(args.verb)
klean = sep.yn2zoo(args.klean)

# Base command for all jobs
bcmd = '/data/sep/joseph29/opt/anaconda3/envs/py35/bin/python ./scripts/tsdata_ptscat_allsep.py -c '

# Create and submit all jobs
actjobs = []; lefjobs = []
for ijob in range(tjobs):
  if(ijob < ajobs):
    # Create job
    args.beg += nex
    actjobs.append(rmtrdat.rmtrjob(args,jobprefix,parpath,logpath,verb=verb))
    cmd = bcmd + actjobs[ijob].pfname
    # Submit job
    actjobs[ijob].submit(jobprefix,cmd,sleep=0.0)
    if(verb): print("Job=%d %s"%(ijob, actjobs[ijob].jobid))
  else:
    # Leftover jobs, to be submitted
    args.beg += nex
    lefjobs.append(rmtrdat.rmtrjob(args,jobprefix,parpath,logpath,verb=verb))

if(verb): print("%d jobs submitted, %d jobs waiting. Managing jobs now...\n"%(len(actjobs),len(lefjobs)))

# Loop until all jobs have completed
while len(actjobs) > 0:
  todel = []
  qlines = pbs.qstat()
  # Check the status of each job
  for ijob in range(len(actjobs)):
    #actjobs[ijob].getstatus()
    actjobs[ijob].getstatus_fast(qlines)
    if(verb):
      print("Job=%d %s sep: %s default: %s"%(ijob, actjobs[ijob].jobid,
        actjobs[ijob].status['sep'], actjobs[ijob].status['default']))
    if('Q' in actjobs[ijob].status.values()):
    # If job is queued, submit to other queue
      cmd = bcmd + actjobs[ijob].pfname
      actjobs[ijob].submit(jobprefix,cmd,sleep=0.0)
      todel.append(False)
    elif('C' in actjobs[ijob].status.values()):
    # If completed delete or resubmit
      if(actjobs[ijob].success('Success!')):
        if(verb): print("Job=%d %s complete!"%(ijob,actjobs[ijob].jobid))
        todel.append(True)
      elif(actjobs[ijob].nsub < args.nsubmit):
        cmd = bcmd + actjobs[ijob].pfname
        actjobs[ijob].submit(jobprefix,cmd,sleep=0.0)
        todel.append(False)
      else:
        if(verb): print("Having trouble submitting job %s. Removing..."%(actjobs[ijob].jobid))
        todel.append(True)
    elif('R' == actjobs[ijob].status['sep'] and 'R' == actjobs[ijob].status['default']):
      actjobs[ijob].cleanup()
      todel.append(False)
    else:
      # Leave the job alone
      todel.append(False)
  # Delete completed jobs
  idx = 0
  for ijob in range(len(actjobs)):
    if(todel[ijob]):
      del actjobs[idx]
    else:
      idx += 1
  # Submit and add leftover jobs
  for ijob in range(len(lefjobs)):
    # Don't append if no active jobs complete
    if(len(actjobs) >= ajobs): break
    cmd = bcmd + lefjobs[0].pfname
    lefjobs[0].submit(jobprefix,cmd,sleep=0.0)
    actjobs.append(lefjobs[0])
    if(verb): print("Submitting waiting Job %s..."%(lefjobs[0].jobid))
    del lefjobs[0]
  if(verb): print("Number of active jobs %d, Number of waiting jobs %d\n"%(len(actjobs),len(lefjobs)))

if(klean):
  # Remove scripts
  rmscr = 'rm %s*.sh'%(jobprefix)
  if(verb): print(rmscr)
  sp = subprocess.check_call(rmscr,shell=True)
  # Remove log files
  rmlog = 'rm %s/%s*.log'%(logpath,jobprefix)
  if(verb): print(rmlog)
  sp = subprocess.check_call(rmlog,shell=True)
  # Remove par files
  rmpar = 'rm %s/%s*.par'%(parpath,jobprefix)
  if(verb): print(rmpar)
  sp = subprocess.check_call(rmpar,shell=True)
  # Remove qstat.out
  rmqs = 'rm qstat.out'
  if(verb): print(rmqs)
  sp = subprocess.check_call(rmqs,shell=True)

