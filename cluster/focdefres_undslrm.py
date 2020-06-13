import cluster.slurmjob as job
from utils.ptyprint import create_inttag

class focdefres_undjob(job.slurmjob):
  """ Keeps track of jobs for creating focused, defocused and residually defocused data """

  def __init__(self,pars,jobname,maxnum=9999,parpath=".",logpath=".",user='joseph29',verb=False):
    # Inherit from job class
    super(focdefres_undjob,self).__init__(logpath,user,verb)
    # Create the par file for this job
    self.pfname = parpath + '/' + jobname + self.jobid + '.par'
    self.write_focdefres_undpar(self.pfname,pars,maxnum)

  def write_focdefres_undpar(self,name,pars,maxnum):
    """ Writes a par file for focused, defocused and residually defocused training data """
    # Build the names
    velname = pars.outdir + '/' + pars.prefix + "vel-" + create_inttag(pars.beg,maxnum) + ".H"
    refname = pars.outdir + '/' + pars.prefix + "ref-" + create_inttag(pars.beg,maxnum) + ".H"
    lblname = pars.outdir + '/' + pars.prefix + "lbl-" + create_inttag(pars.beg,maxnum) + ".H"
    cnvname = pars.outdir + '/' + pars.prefix + "cnv-" + create_inttag(pars.beg,maxnum) + ".H"
    ptbname = pars.outdir + '/' + pars.prefix + "ptb-" + create_inttag(pars.beg,maxnum) + ".H"
    fogname = pars.outdir + '/' + pars.prefix + "fog-" + create_inttag(pars.beg,maxnum) + ".H"
    fagname = pars.outdir + '/' + pars.prefix + "fag-" + create_inttag(pars.beg,maxnum) + ".H"
    dogname = pars.outdir + '/' + pars.prefix + "dog-" + create_inttag(pars.beg,maxnum) + ".H"
    dagname = pars.outdir + '/' + pars.prefix + "dag-" + create_inttag(pars.beg,maxnum) + ".H"
    rogname = pars.outdir + '/' + pars.prefix + "rog-" + create_inttag(pars.beg,maxnum) + ".H"
    ragname = pars.outdir + '/' + pars.prefix + "rag-" + create_inttag(pars.beg,maxnum) + ".H"
    # Build the par file
    parout="""[defaults]
# IO
vel=%s
ptb=%s
ref=%s
lbl=%s
cnv=%s
fimgo=%s
fimga=%s
dimgo=%s
dimga=%s
rimgo=%s
rimga=%s
dpath=%s
# Windowing
na=%d
nro=%d
oro=%f
dro=%f
offset=%d
"""%(velname,ptbname,refname,lblname,cnvname,                                    # Output inputs
     fogname,fagname,dogname,dagname,rogname,ragname,pars.dpath,                 # Output outputs
    int(pars.na),int(pars.nro),float(pars.oro),float(pars.dro),int(pars.offset)) # Angles and resmig
    # Write the par file
    with open(name,'w') as f:
      f.write(parout)

    return

