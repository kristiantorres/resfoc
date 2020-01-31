import cluster.slurmjob as job

class veljob(job.slurmjob):
  """ Keeps track of jobs for creating velocity models """

  def __init__(self,pars,jobname,parpath=".",logpath=".",user='joseph29',verb=False):
    # Inherit from job class
    super(veljob,self).__init__(logpath,user,verb)
    # Create the par file for this job
    self.pfname = parpath + '/' + jobname + self.jobid + '.par'
    self.write_velpar(self.pfname,pars)
    # Keep the beginning example number
    self.beg = pars.beg

  def write_velpar(self,name,pars):
    """Writes a par file for residual migration training data """
    # Build the par file
    parout="""[defaults]
# IO
outdir=%s
datapath=%s
beg=%d
end=%d
prefix=%s
nmodels=%d
# Velocity model
nx=%d
ox=%f
dx=%f
ny=%d
oy=%f
dy=%f
nz=%d
oz=%f
dz=%f
# Processing
nzo=%d
nxo=%d
rect=%f
# Other
verb=%s
"""%(pars.outdir, pars.datapath, pars.beg, pars.end, pars.prefix, pars.nmodels, #IO
    pars.nx, pars.ox, pars.dx, pars.ny, pars.oy, pars.dy, pars.nz, pars.oz, pars.dz, # Velocity
    pars.nzo, pars.nxo, pars.rect, # Processing
    pars.verb) #Other
    # Write the par file
    with open(name,'w') as f:
      f.write(parout)

    return

