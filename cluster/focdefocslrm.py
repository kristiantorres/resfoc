import cluster.slurmjob as job
from utils.ptyprint import create_inttag

class focdefocjob(job.slurmjob):
  """ Keeps track of jobs for creating refocusing training data """

  def __init__(self,pars,jobname,maxnum=9999,parpath=".",logpath=".",user='joseph29',verb=False):
    # Inherit from job class
    super(focdefocjob,self).__init__(logpath,user,verb)
    # Create the par file for this job
    self.pfname = parpath + '/' + jobname + self.jobid + '.par'
    self.write_focdefocpar(self.pfname,pars,maxnum)

  def write_focdefocpar(self,name,pars,maxnum):
    """ Writes a par file for focus defocus training data """
    # Build the names
    velname = pars.outdir + '/' + pars.prefix + "vel-" + create_inttag(pars.beg,maxnum) + ".H"
    refname = pars.outdir + '/' + pars.prefix + "ref-" + create_inttag(pars.beg,maxnum) + ".H"
    lblname = pars.outdir + '/' + pars.prefix + "lbl-" + create_inttag(pars.beg,maxnum) + ".H"
    ptbname = pars.outdir + '/' + pars.prefix + "ptb-" + create_inttag(pars.beg,maxnum) + ".H"
    fogname = pars.outdir + '/' + pars.prefix + "fog-" + create_inttag(pars.beg,maxnum) + ".H"
    fagname = pars.outdir + '/' + pars.prefix + "fag-" + create_inttag(pars.beg,maxnum) + ".H"
    dogname = pars.outdir + '/' + pars.prefix + "dog-" + create_inttag(pars.beg,maxnum) + ".H"
    dagname = pars.outdir + '/' + pars.prefix + "dag-" + create_inttag(pars.beg,maxnum) + ".H"
    # Build the par file
    parout="""[defaults]
# IO
vel=%s
ptb=%s
ref=%s
lbl=%s
fimgo=%s
fimga=%s
dimgo=%s
dimga=%s
dpath=%s
# Windowing
fx=%d
nxw=%d
na=%d
"""%(velname,ptbname,refname,lblname,fogname,fagname,dogname,dagname,pars.dpath, #IO
    int(pars.fx),int(pars.nxw),int(pars.na)) # Windowing
    # Write the par file
    with open(name,'w') as f:
      f.write(parout)

    return

