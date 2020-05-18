import cluster.slurmjob as job
from utils.ptyprint import create_inttag

class resdefocjob(job.slurmjob):
  """ Keeps track of jobs for creating refocusing training data """

  def __init__(self,pars,jobname,maxnum=9999,parpath=".",logpath=".",user='joseph29',verb=False):
    # Inherit from job class
    super(resdefocjob,self).__init__(logpath,user,verb)
    # Create the par file for this job
    self.pfname = parpath + '/' + jobname + self.jobid + '.par'
    self.write_resdefocpar(self.pfname,pars,maxnum)

  def write_resdefocpar(self,name,pars,maxnum):
    """ Writes a par file for residually defocused training data """
    # Build the names
    fogname  = pars.inpdir + '/' + pars.iprefix + "fog-"  + create_inttag(pars.beg,maxnum) + ".H"
    resoname = pars.outdir + '/' + pars.oprefix + "reso-" + create_inttag(pars.beg,maxnum) + ".H"
    resaname = pars.outdir + '/' + pars.oprefix + "resa-" + create_inttag(pars.beg,maxnum) + ".H"
    # Build the par file
    parout="""[defaults]
# IO
img=%s
reso=%s
resa=%s
dpath=%s
# Windowing
nro=%d
oro=%f
dro=%f
na=%d
verb=y
"""%(fogname,resoname,resaname,pars.dpath,# IO
     int(pars.nro),float(pars.oro),float(pars.dro),int(pars.na)) # Residual migration
    # Write the par file
    with open(name,'w') as f:
      f.write(parout)

    return

