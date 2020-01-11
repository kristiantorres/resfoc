# Helper functions for submitting jobs to a PBS cluster
import os
import subprocess
import string, random
import time

def write_script(name,cmd,nprocs=8,queue='sep',logpath='.',outfile=None,errfile=None,erase=True):
  """ Writes a slurm script to file """
  if(outfile == None):
    outfile = name
  if(errfile == None):
    errfile = name
  # Check if outfile and errfile exist
  tag = ""
  if(os.path.exists(outfile+"_out.log") and erase==True):
    sp = subprocess.check_call('rm %s'%(logpath+'/'+outfile+"_out.log"),shell=True)
  elif(os.path.exists(outfile) and erase==False):
    tag = id_generator()
  outfile = logpath + '/' + outfile + tag + "_out.log"
  errfile = logpath + '/' + errfile + tag + "_err.log"
  # Create the PBS script
  slurmout = """#! /bin/tcsh
#SBATCH --job-name %s
#SBATCH --ntasks=1 nodes=1:ppn=%d
#SBATCH --cpus-per-task=%d
#SBATCH --partition=%s
#SBATCH --time=02:00:00
#SBATCH --output=%s
#SBATCH --error=%s
cd $SLURM_SUBMIT_DIR
#
%s
#
# End of script"""%(name,nprocs,queue,outfile,errfile,cmd)
  with open(name+".sh",'w') as f:
    f.write(slurmout)

  return name+".sh"

def submit_job(script,sleep=1,submit=True,verb=True):
  """ Prints and submits a job to a PBS cluster """
  sub = "sbatch %s"%(script)
  if(submit):
    if(verb): print(sub)
    time.sleep(sleep)
    sp = subprocess.check_call(sub,shell=True)
  else:
    print(sub)

  return

def get_numjobs(queue,user='joseph29',qfile=None):
  """ Gets the number of running and completed jobs of a user in a specific queue """
  # Initialize the dictionary
  numjobs = {}
  numjobs['R'] = 0; numjobs['C'] = 0; numjobs['Q'] = 0
  # Change the queue to have spaces before and after
  queue = ' ' + queue + ' '
  # Get the output of qstat -u user
  if(qfile == None):
    ujobs = "qstat -u %s > qstat.out"%(user)
    sp = subprocess.check_call(ujobs,shell=True)
    # Read in the file
    with open('qstat.out', 'r') as f:
      for line in f.readlines():
        if(user in line and queue in line and ' R ' in line):
          numjobs['R'] += 1
        elif(user in line and queue in line and ' Q ' in line):
          numjobs['Q'] += 1
        elif(user in line and queue in line and ' H ' in line):
          numjobs['Q'] += 1
        elif(user in line and queue in line and ' C ' in line):
          numjobs['C'] += 1
        elif(user in line and queue in line and ' E ' in line):
          numjobs['C'] += 1
  else:
    for line in qfile:
      if(user in line and queue in line and ' R ' in line):
        numjobs['R'] += 1
      elif(user in line and queue in line and ' Q ' in line):
        numjobs['Q'] += 1
      elif(user in line and queue in line and ' H ' in line):
        numjobs['Q'] += 1
      elif(user in line and queue in line and ' C ' in line):
        numjobs['C'] += 1
      elif(user in line and queue in line and ' E ' in line):
        numjobs['C'] += 1

  return numjobs

def killjobs(user='joseph29',queue=None,state='Q'):
  """ Kills the jobs for a specific user in a specific queue for a specific state """
  # First run qstat
  ujobs = 'qstat -u %s > qstat.out'%(user)
  sp = subprocess.check_call(ujobs,shell=True)
  with open('qstat.out', 'r') as f:
    for line in f.readlines():
      if(user in line):
        attrs = line.split()
        if(None == queue and None == state):
          subid = attrs[0].split('.')[0]
          sp = subprocess.check_call('scancel %s'%(subid),shell=True)
        elif(attrs[2] == queue and attrs[9] == state):
          subid = attrs[0].split('.')[0]
          sp = subprocess.check_call('scancel %s'%(subid),shell=True)
        elif(None == queue and attrs[9] == state):
          subid = attrs[0].split('.')[0]
          sp = subprocess.check_call('scancel %s'%(subid),shell=True)
        elif(attrs[2] == queue):
          subid = attrs[0].split('.')[0]
          sp = subprocess.check_call('scancel %s'%(subid),shell=True)

def qstat(user='joseph29'):
  """ Gets the output of qstat from python """
  ujobs = 'qstat -u %s > qstat.out'%(user)
  sp = subprocess.check_call(ujobs,shell=True)
  with open('qstat.out', 'r') as f:
    return f.readlines()

def squeue():
  """ Gets the output of squeue from python """
  squeue = 'squeue -o "%.18i %.9P %.15j %.8u %.2t %.10M %.6D %R" > squeue.out'
  sp = subprocess.check_call(squeue,shell=True)
  with open('squeue.out', 'r') as f:
    return f.readlines()

def id_generator(self,size=6, chars=string.ascii_uppercase + string.digits):
  """ Creates a random string with uppercase letters and integers """
  return ''.join(random.choice(chars) for _ in range(size))

