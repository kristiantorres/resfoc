import cluster.slurmjob as job

class resmigjob(job.slurmjob):
  """ Keeps track of jobs for performing residual migration """

  def __init__(self,pars,velname,refname,idx,jobname,parpath=".",logpath=".",user='joseph29',verb=False):
    # Inherit from job class
    super(resmigjob,self).__init__(logpath,user,verb)
    # Create the par file for this job
    self.pfname = parpath + '/' + jobname + self.jobid + '.par'
    self.write_resmigpar(self.pfname,pars,velname,refname,idx)
    # Keep names and idx for this job
    self.velname = velname; self.refname = refname
    self.idx = idx

  def write_resmigpar(self,name,pars,velname,refname,idx):
    """Writes a par file for residual migration fault training data """
    # Build the par file
    parout="""[defaults]
# IO
velin=%s
refin=%s
outdir=%s
dpath=%s
ptbpf=%s
imgpf=%s
velidx=%d
# Other
verb=%s
nthreads=24
"""%(velname, refname, pars.outdir, pars.datapath, pars.ptbpf, pars.imgpf, idx, #IO
     pars.verb) # Other
    # Write the par file
    with open(name,'w') as f:
      f.write(parout)

    return

