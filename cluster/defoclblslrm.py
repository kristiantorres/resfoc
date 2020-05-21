import cluster.slurmjob as job
from utils.ptyprint import create_inttag

class defoclbljob(job.slurmjob):
  """ Keeps track of jobs for labeling defocused image patches """

  def __init__(self,pars,jobname,maxnum=9999,parpath=".",logpath=".",user='joseph29',verb=False):
    # Inherit from job class
    super(defoclbljob,self).__init__(logpath,user,verb)
    # Create the par file for this job
    self.pfname = parpath + '/' + jobname + self.jobid + '.par'
    self.write_defoclblpar(self.pfname,pars,maxnum)

  def write_defoclblpar(self,name,pars,maxnum):
    """ Writes a par file for labeling defocused images data """
    # Build the names
    deflbls = pars.oprefix + str(pars.begex) + ".h5"
    endex = pars.begex + pars.nex
    # Build the par file
    parout="""[defaults]
# IO
focptch=%s
defptch=%s
focprb=%s
defprb=%s
deflbls=%s
begex=%d
endex=%d
# Labeling
verb=y
qcplot=n
nqc=0
thresh1=%f
thresh2=%f
"""%(pars.focptch,pars.defptch,pars.focprb,pars.defprb,deflbls,pars.begex,endex,# IO
     float(pars.thresh1),float(pars.thresh2)) # Labeling
    # Write the par file
    with open(name,'w') as f:
      f.write(parout)

    return

