"""
Creates many residually migrated defocused images
and computes semblances and the angle stack on these images

@author: Joseph Jennings
@version: 2020.05.02
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from cluster.refocslrm import refocjob
import cluster.slurmhelper as slurm
import time
import subprocess, glob

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "outdir": "/data/sep/joseph29/projects/resfoc/bench/dat/refoc",
    "prefix": "refoc",
    "dpath": "/data/sep/joseph29/scratch/refoc",
    "fx": 373,
    "nxw": 512,
    "tjobs": 1000,
    "ajobs": 200,
    "nprocs": 16,
    "nsubmit": 5,
    "logpath": "./log/refoc",
    "parpath": "./par/refoc",
    "jobprefix": "refoc",
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
winArgs.add_argument("-fx",help="First x sample [373]",type=int)
winArgs.add_argument("-nxw",help="Length of window in x [512]",type=int)
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
bcmd = '/data/biondo/joseph29/opt/anaconda3/envs/py35/bin/python ./scripts/layered_fault_refocus.py -c '

# Create and submit all jobs
sepqfull = False; hr2qfull= False
actjobs = []; lefjobs = []
# Starting queue
squeue = 'sep'
for ijob in range(tjobs):
  if(ijob < ajobs):
    # Create job
    actjobs.append(refocjob(args,jobprefix,maxnum,parpath,logpath,verb=verb))
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
    lefjobs.append(refocjob(args,jobprefix,maxnum,parpath,logpath,verb=verb))

if(verb): print("%d jobs submitted, %d jobs waiting. Managing jobs now...\n"%(len(actjobs),len(lefjobs)))

# Loop until all jobs have completed
while len(actjobs) > 0:
  todel = []
  # First update qstat and squeue
  qlines = slurm.qstat(); qqueue = slurm.squeue()
  sepq = slurm.get_numjobs('sep',qfile=qlines)
  hr2q = slurm.get_numjobs('twohour',qfile=qlines)
  # Check the status of each job
  for ijob in range(len(actjobs)):
    actjobs[ijob].getstatus_fast(qlines,qqueue)
    if(verb):
      print("Job=%d %s sep: %s %s twohour: %s %s"%(ijob, actjobs[ijob].jobid,
        actjobs[ijob].status['sep'],     actjobs[ijob].nodes['sep'],
        actjobs[ijob].status['twohour'], actjobs[ijob].nodes['twohour']))
    if(actjobs[ijob].status['sep'] == None and actjobs[ijob].status['twohour'] == None):
      # Resubmit if None and None
      if(sepq['Q'] < nleft):
        cmd = bcmd + actjobs[ijob].pfname
        actjobs[ijob].submit(jobprefix,cmd,nprocs=nprocs,queue='sep',sleep=0.5)
        if(verb): print("Resubmitting stale Job=%d %s to queue sep..."%(ijob,actjobs[ijob].jobid))
      elif(hr2q['Q'] < nleft):
        cmd = bcmd + actjobs[ijob].pfname
        actjobs[ijob].submit(jobprefix,cmd,nprocs=nprocs,queue='twohour',sleep=0.5)
        if(verb): print("Resubmitting stale Job=%d %s to queue twohour..."%(ijob,actjobs[ijob].jobid))
      todel.append(False)
    if('C' in actjobs[ijob].status.values()):
      # If completed delete or resubmit
      if(actjobs[ijob].success('Success!')):
        if(verb): print("Job=%d %s complete!"%(ijob,actjobs[ijob].jobid))
        todel.append(True)
      elif(actjobs[ijob].nsub < args.nsubmit):
        # First, get the queue of the job
        myqueue = list(actjobs[ijob].status.keys())[list(actjobs[ijob].status.values()).index('C')]
        # Check on what node the job failed
        if(verb):
          print("Job=%d %s failed %d times on node %s"%(ijob, actjobs[ijob].jobid, actjobs[ijob].nsub, actjobs[ijob].nodes[myqueue]))
        if(sepq['Q'] < nleft):
          cmd = bcmd + actjobs[ijob].pfname
          actjobs[ijob].submit(jobprefix,cmd,nprocs=nprocs,queue='sep',sleep=0.5)
          if(verb): print("Resubmitting failed Job=%d %s to queue sep..."%(ijob,actjobs[ijob].jobid))
        elif(hr2q['Q'] < nleft):
          cmd = bcmd + actjobs[ijob].pfname
          actjobs[ijob].submit(jobprefix,cmd,nprocs=nprocs,queue='twohour',sleep=0.5)
          if(verb): print("Resubmitting failed Job=%d %s to queue twohour..."%(ijob,actjobs[ijob].jobid))
        todel.append(False)
      else:
        if(verb): print("Submitted Job=%d %s %d times and failed each time. Removing..."%(ijob,actjobs[ijob].jobid,args.nsubmit))
        if(actjobs[ijob].status['sep'] == 'Q'):
          actjobs[ijob].delete('sep')
        if(actjobs[ijob].status['twohour'] == 'Q'):
          actjobs[ijob].delete('twohour')
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
    cmd = bcmd + lefjobs[0].pfname
    # Don't submit to a queue that is full
    if(sepq['Q'] < 2):
      lefjobs[0].submit(jobprefix,cmd,nprocs=nprocs,queue='sep',sleep=0.5)
      actjobs.append(lefjobs[0])
      if(verb): print("Submitting waiting Job %s to sep queue..."%(lefjobs[0].jobid))
      del lefjobs[0]
    elif(hr2q['Q'] < 2):
      lefjobs[0].submit(jobprefix,cmd,nprocs=nprocs,queue='twohour',sleep=0.5)
      actjobs.append(lefjobs[0])
      if(verb): print("Submitting waiting Job %s to twohour queue..."%(lefjobs[0].jobid))
      del lefjobs[0]
    else:
      sepqfull = True; hr2qfull = True
      if(verb): print("Both sep and twohour queues are full. Not submitting for now...")
  # Make sure that at least two are waiting in both queues
  sepq = slurm.get_numjobs('sep',qfile=qlines)
  #TODO: remove the sepqfull and add a while loop while (nleft - sepq['Q'] - 1 > 0) and update sepq each time
  if(sepqfull and sepq['Q'] < nleft and len(lefjobs) > 0):
    for ijob in range(nleft - sepq['Q'] - 1):
      cmd = bcmd + lefjobs[0].pfname
      lefjobs[0].submit(jobprefix,cmd,nprocs=nprocs,queue='sep',sleep=0.5)
      actjobs.append(lefjobs[0])
      if(verb): print("Submitting waiting Job %s in sep for queuing..."%(lefjobs[0].jobid))
      del lefjobs[0]
  hr2q = slurm.get_numjobs('twohour',qfile=qlines)
  if(hr2qfull and hr2q['Q'] < nleft and len(lefjobs) > 0):
    for ijob in range(nleft - hr2q['Q'] - 1):
      cmd = bcmd + lefjobs[0].pfname
      lefjobs[0].submit(jobprefix,cmd,nprocs=nprocs,queue='twohour',sleep=0.5)
      actjobs.append(lefjobs[0])
      if(verb): print("Submitting waiting Job %s in twohour for queuing..."%(lefjobs[0].jobid))
      del lefjobs[0]
  if(verb):
    print("Number of active jobs %d, Number of waiting jobs %d"%(len(actjobs),len(lefjobs)))
    print("sep queue: %d R %d Q %d C"%(sepq['R'],sepq['Q'],sepq['C']))
    print("2hr queue: %d R %d Q %d C\n"%(hr2q['R'],hr2q['Q'],hr2q['C']))

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
  # Remove node files
  rmnds = 'rm *-node.txt'
  if(verb): print(rmnds)
  sp = subprocess.check_call(rmnds,shell=True)
  # Remove squeue.out
  rmsq = 'rm squeue.out'
  if(verb): print(rmsq)
  sp = subprocess.check_call(rmsq,shell=True)
  # Remove qstat.out
  rmqs = 'rm qstat.out'
  if(verb): print(rmqs)
  sp = subprocess.check_call(rmqs,shell=True)

