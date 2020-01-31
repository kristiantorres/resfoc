import clusterhelp.pbsjob as job

class rmtrjob(job.pbsjob):
  """ Keeps track of jobs for the residual migration training data generation """

  def __init__(self,pars,jobname,parpath=".",logpath=".",user='joseph29',verb=False):
    # Inherit from job class
    super(rmtrjob,self).__init__(logpath,user,verb)
    # Create the par file for this job
    self.pfname = parpath + '/' + jobname + self.jobid + '.par'
    self.write_trdatpar(self.pfname,pars)
    # Keep the beginning example number
    self.beg = pars.beg

  def write_trdatpar(self,name,pars):
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
osx=%d
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

    return

