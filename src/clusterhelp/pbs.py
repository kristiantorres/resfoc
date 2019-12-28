# Helper functions for submitting jobs to a PBS cluster
import subprocess as sp

def write_trdatpar(name,pars):
  """Writes a par file for residual migration training data """
  # Build the par file
  parout="""[defaults]
# IO
outdir=%s
datapath=%s
nwrite=%d
beg=%d
end=%d
prefix=%s
# Imaging parameters
nsx=%d
osx=%f
nz=%d
nx=%d
nh=%d
# Residual migration
nro=%d
oro=%f
dro=%f
# ML parameters
nex=%d
# Other
verb=%s
nprint=%d
"""%(pars.outdir,pars.datapath,pars.nwrite,pars.beg,pars.end,pars.prefix, #IO
    pars.nsx,pars.osx,pars.nz,pars.nx,pars.nh, # Imaging
    pars.nro,pars.oro,pars.dro, # Residual migration
    pars.nex, # ML
    pars.verb,pars.nprint) #Other
  # Write the par file
  with open(name,'w') as f:
    f.write(parout)

def write_script(name,cmd,nprocs=8,queue='sep',outfile=None,errfile=None):
  """ Writes a PBS script to file """
  if(outfile == None):
    outfile = name + "_out.log"
  if(errfile == None):
    errfile = name + "_err.log"
  pbsout = """#! /bin/tcsh

#PBS -N %s
#PBS -l nodes=1:ppn=%d
#PBS -q %s
#PBS -o %s
#PBS -e %s
cd $PBS_O_WORKDIR
#
%s
#
# End of script"""%(name,nprocs,queue,outfile,errfile,cmd)
  with open(name+".sh",'w') as f:
    f.write(pbsout)

  return name+".sh"

def submit_job(script,submit=True,verb=True):
  """ Prints and submits a job to a PBS cluster """
  sub = "qsub %s"%(script)
  if(submit):
    if(verb): print(sub)
    sp = subprocess.check_call(sub,shell=True)
  else:
    print(sub)

def get_numjobs(user,queue):
  """ Gets the number of jobs of a user in a specific queue """
  # Get the output of qstat -u user
  ujobs = "qstat -u %s > qstat.out"%(user)
  sp = subprocess.check_call(ujobs,shell=True)
  # Read in the file
  numjobs = 0
  with open('qstat.out', 'r') as f:
    for line in f.readlines():
      if(user in line and queue in line):
        numjobs += 1

  return numjobs

