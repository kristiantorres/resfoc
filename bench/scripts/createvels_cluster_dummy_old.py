"""
Creates many random but geologically feasible velocity models
using software developed by Bob clapp and a PBS/Torque cluster.
to a PBS/Torque cluster. By default, the script attempts run 200 jobs
(total examples = no. of jobs * nexamples per job). The user can
override these parameters in the cluster arguments section

@author: Joseph Jennings
@version: 2020.01.07
"""

import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import cluster.velbuild as velbuild
import cluster.pbshelper as pbs
import time
import subprocess

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "nx": 600,
    "ox": 0.0,
    "dx": 25.0,
    "ny": 600,
    "oy": 0.0,
    "dy": 25.0,
    "nz": 600.0,
    "oz": 0.0,
    "dz": 10.0,
    "rect": 0.8,
    "nmodels": 1,
    "prefix": "",
    "beg": 0,
    "end": 9999,
    "tjobs": 600,
    "ajobs": 200,
    "nsubmit": 5,
    "logpath": ".",
    "parpath": ".",
    "jobprefix": "velmod",
    "nleft": 2,
    "delay": 10.0,
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
ioArgs.add_argument("-outdir",help="Output directory of where to write the data and par files",type=str)
ioArgs.add_argument("-datapath",help="Output datapath of where to write the SEPlib binaries",type=str)
ioArgs.add_argument("-prefix",help="Prefix that will be used for data and par files [None]",type=str)
ioArgs.add_argument("-beg",help="Numeric suffix used for keeping track of examples [0]",type=int)
ioArgs.add_argument("-end",help="Last example for writing [9999]",type=int)
ioArgs.add_argument("-nmodels",help="Number of velocity models to write [10]",type=int)
# Velocity parameters
velArgs = parser.add_argument_group("Velocity model parameters")
velArgs.add_argument("-nx",help="Number of x samples [600]",type=int)
velArgs.add_argument("-ox",help="x origin [0.0]",type=float)
velArgs.add_argument("-dx",help="x sample spacing [25.0]",type=float)
velArgs.add_argument("-ny",help="Number of y samples [600]",type=int)
velArgs.add_argument("-oy",help="y origin [0.0]",type=float)
velArgs.add_argument("-dy",help="y sample spacing [25.0]",type=float)
velArgs.add_argument("-nz",help="Number of depth samples [600]",type=int)
velArgs.add_argument("-oz",help="z origin [0.0]",type=float)
velArgs.add_argument("-dz",help="z sample spacing [10.0]",type=float)
# Model processing parameters
prcArgs = parser.add_argument_group("Velocity model processing")
prcArgs.add_argument("-nzo",help="Ouput number of depth samples for interpolation [256]",type=int)
prcArgs.add_argument("-nxo",help="Ouput number of lateral samples for interpolation [400]",type=int)
prcArgs.add_argument("-rect",help="Window radius for smoother [0.5]",type=float)
# Other arguments
othArgs = parser.add_argument_group('Other parameters')
othArgs.add_argument("-verb",help="Verbosity flag [y]",type=str)
# Cluster arguments
cluArgs = parser.add_argument_group('Cluster parameters')
cluArgs.add_argument("-tjobs",help="Total number of jobs to run [600]",type=int)
cluArgs.add_argument("-ajobs",help="Number of jobs either in queue or running at once [200], cannot do more than 200 in twohour queue",type=int)
cluArgs.add_argument("-nsubmit",help="Number of times to attempt a job submission [5]",type=int)
cluArgs.add_argument("-logpath",help="Path to logfile [current directory]",type=str)
cluArgs.add_argument("-parpath",help="Path to parfile [current directory]",type=str)
cluArgs.add_argument("-jobprefix",help="Job prefix for par files [velmod]",type=str)
cluArgs.add_argument("-nleft",help="Number of jobs to be queued in each queue [2]",type=int)
cluArgs.add_argument("-delay",help="Amount of time in seconds to wait between prints [10]",type=float)
cluArgs.add_argument("-klean",help="Clean up cluster submission files [y]",type=str)
args = parser.parse_args(remaining_argv)

# Setup IO
sep = seppy.sep(sys.argv)

# Get command line arguments
tjobs = args.tjobs; ajobs = args.ajobs
nmodels = args.nmodels
logpath = args.logpath; parpath = args.parpath
jobprefix = args.jobprefix
verb = sep.yn2zoo(args.verb);
klean = sep.yn2zoo(args.klean)
nleft = args.nleft
delay = args.delay

# Base command for all jobs
#bcmd = '/data/sep/joseph29/opt/anaconda3/envs/py35/bin/python ./scripts/Create_randomvel.py -c '
cmd = """sleep 300
echo 'Success!'"""

# Create and submit all jobs
sepqfull = False; hr2qfull = False
actjobs = []; lefjobs = []
# Starting queue
squeue = 'sep'
for ijob in range(tjobs):
  if(ijob < ajobs):
    # Create job
    args.beg += nmodels
    actjobs.append(velbuild.veljob(args,jobprefix,parpath,logpath,verb=verb))
    # Submit job
    actjobs[ijob].submit(jobprefix,cmd,queue=squeue,sleep=3.0)
    # Get the status of the queues
    qlines = pbs.qstat()
    sepq = pbs.get_numjobs('sep',qfile=qlines)
    hr2q = pbs.get_numjobs('twohour',qfile=qlines)
    if(verb):
      print("sep queue: %d R %d Q %d C"%(sepq['R'],sepq['Q'],sepq['C']))
      print("2hr queue: %d R %d Q %d C"%(hr2q['R'],hr2q['Q'],hr2q['C']))
    if(sepq['Q'] >= 2):
      squeue = 'twohour'
      sepqfull = True
    elif(hr2q['Q'] >= 2):
      ajobs = sepq['R'] + hr2q['R']
      hr2qfull = True
    if(verb): print("Job=%d %s"%(ijob, actjobs[ijob].jobid))
  else:
    # Leftover jobs, to be submitted
    args.beg += nmodels
    lefjobs.append(velbuild.veljob(args,jobprefix,parpath,logpath,verb=verb))

if(verb): print("%d jobs submitted, %d jobs waiting. Managing jobs now...\n"%(len(actjobs),len(lefjobs)))

# Loop until all jobs have completed
while len(actjobs) > 0:
  todel = []
  qlines = pbs.qstat()
  sepq = pbs.get_numjobs('sep',qfile=qlines)
  hr2q = pbs.get_numjobs('twohour',qfile=qlines)
  # Check the status of each job
  for ijob in range(len(actjobs)):
    actjobs[ijob].getstatus_fast(qlines)
    if(verb):
      print("Job=%d %s sep: %s twohour: %s"%(ijob, actjobs[ijob].jobid,
        actjobs[ijob].status['sep'], actjobs[ijob].status['twohour']))
    if('C' in actjobs[ijob].status.values()):
    # If completed delete or resubmit
      if(actjobs[ijob].success('Success!')):
        if(verb): print("Job=%d %s complete!"%(ijob,actjobs[ijob].jobid))
        todel.append(True)
      elif(actjobs[ijob].nsub < args.nsubmit):
        actjobs[ijob].submit(jobprefix,cmd,sleep=2.5)
        todel.append(False)
      else:
        if(verb): print("Having trouble submitting job %s. Removing..."%(actjobs[ijob].jobid))
        todel.append(True)
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
    lefjobs[0].submit(jobprefix,cmd,sleep=2.5)
    actjobs.append(lefjobs[0])
    if(verb): print("Submitting waiting Job %s..."%(lefjobs[0].jobid))
    del lefjobs[0]
  # Make sure that at least two are waiting in both queues
  sepqdiff = pbs.get_numjobs('sep',qfile=qlines)['Q'] - nleft
  if(sepqfull and sepqdiff > 0 and len(lefjobs) > 0):
    for ijob in range(sepqdiff):
      lefjobs[0].submit(jobprefix,cmd,sleep=2.5)
      actjobs.append(lefjobs[0])
      if(verb): print("Submitting waiting Job %s for queuing..."%(lefjobs[0].jobid))
      del lefjobs[0]
  hr2qdiff = pbs.get_numjobs('twohour',qfile=qlines)['Q'] - nleft
  if(hr2qfull and hr2qdiff > 0 and len(lefjobs) > 0):
    for ijob in range(sepqdiff):
      lefjobs[0].submit(jobprefix,cmd,sleep=2.5)
      actjobs.append(lefjobs[0])
      if(verb): print("Submitting waiting Job %sf for queuing..."%(lefjobs[0].jobid))
      del lefjobs[0]
  if(verb): print("Number of active jobs %d, Number of waiting jobs %d\n"%(len(actjobs),len(lefjobs)))
  time.sleep(delay)

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
