"""
A job manager for a slurm cluster

@author: Joseph Jennings
@version: 2020.05.11
"""
import cluster.slurmhelper as slurm
import subprocess, glob
import time

def manage_slurmjobs(actjobs,bcmd,jobprefix,lefjobs,ajobs,nprocs=48,nleft=2,nsubmit=5,
                     sepqfull=False,hr2qfull=False,delay=5.0,verb=True):
  """
  A job manager for a slurm cluster

  Parameters
    actjobs  - a list of job objects that have been submitted to the
              cluster
    lefjobs  - a list of job objects that have yet to be submitted
    ajobs    - number of active jobs to maitain running
    logpath  - path to logfiles
    parpath  - path to par files
    sepqfull - flag indicating whether the sep queue is full
    hr2qfull - flag indicating whether the two hour queue is full
    delay    - delay between checking all jobs [5.0 seconds]
    verb     - verbosity flag [True]
  """
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
        elif(actjobs[ijob].nsub < nsubmit):
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
          if(verb): print("Submitted Job=%d %s %d times and failed each time. Removing..."%(ijob,actjobs[ijob].jobid,nsubmit))
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
    #TODO: I need to update the queues inside this loop
    # Make sure that at least two are waiting in both queues
    sepq = slurm.get_numjobs('sep',qfile=qlines)
    #TODO: remove the sepqfull and add a while loop while (nleft - sepq['Q'] - 1 > 0) and update sepq each time
    # Ensure that at least nleft jobs are waiting in the queue
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

def clean_clusterfiles(jobprefix,logpath,parpath,verb=True):
  """
  Cleans the files left behind from cluster submission

  Parameters:
    jobprefix - jobprefix for all files
    logpath   - Path to the logfiles
    parpath   - Path to the parfiles
    verb      - verbosity flag [True]
  """
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

