import cluster.slurmjob as job
from utils.ptyprint import create_inttag

class refocjob(job.slurmjob):
  """ Keeps track of jobs for creating refocusing training data """

  def __init__(self,pars,jobname,maxnum=9999,parpath=".",logpath=".",user='joseph29',verb=False):
    # Inherit from job class
    super(refocjob,self).__init__(logpath,user,verb)
    # Create the par file for this job
    self.pfname = parpath + '/' + jobname + self.jobid + '.par'
    self.write_refocpar(self.pfname,pars,maxnum)

  def write_refocpar(self,name,pars,maxnum):
    """ Writes a par file for migration fault training data """
    # Build the names
    velname = pars.outdir + '/' + pars.prefix + "vel-" + create_inttag(pars.beg,maxnum) + ".H"
    refname = pars.outdir + '/' + pars.prefix + "ref-" + create_inttag(pars.beg,maxnum) + ".H"
    lblname = pars.outdir + '/' + pars.prefix + "lbl-" + create_inttag(pars.beg,maxnum) + ".H"
    ptbname = pars.outdir + '/' + pars.prefix + "ptb-" + create_inttag(pars.beg,maxnum) + ".H"
    resname = pars.outdir + '/' + pars.prefix + "res-" + create_inttag(pars.beg,maxnum) + ".H"
    stkname = pars.outdir + '/' + pars.prefix + "stk-" + create_inttag(pars.beg,maxnum) + ".H"
    smbname = pars.outdir + '/' + pars.prefix + "smb-" + create_inttag(pars.beg,maxnum) + ".H"
    # Build the par file
    parout="""[defaults]
# IO
vel=%s
ref=%s
lbl=%s
ptb=%s
res=%s
stk=%s
smb=%s
dpath=%s
# Windowing
fx=%d
nxw=%d
"""%(velname,refname,lblname,ptbname,resname,stkname,smbname,pars.dpath, #IO
    int(pars.fx),int(pars.nxw)) # Windowing
    # Write the par file
    with open(name,'w') as f:
      f.write(parout)

    return

