import cluster.slurmjob as job
from utils.ptyprint import create_inttag

class defocanglbljob(job.slurmjob):
  """ Keeps track of jobs for labeling defocused angle gather image patches """

  def __init__(self,pars,jobname,maxnum=9999,parpath=".",logpath=".",user='joseph29',verb=False):
    # Inherit from job class
    super(defocanglbljob,self).__init__(logpath,user,verb)
    # Create the par file for this job
    self.pfname = parpath + '/' + jobname + self.jobid + '.par'
    self.write_defocanglblpar(self.pfname,pars,maxnum)

  def write_defocanglblpar(self,name,pars,maxnum):
    """ Writes a par file for labeling defocused images data """
    # Build the names
    deflbls = pars.oprefix + str(pars.begex) + ".h5"
    endex = pars.begex + pars.nex
    if(endex > pars.totex): endex = pars.totex
    # Build the par file
    parout="""[defaults]
# IO
focptch=%s
defptch=%s
fltptch=%s
focprb=%s
defprb=%s
deflbls=%s
begex=%d
endex=%d
# Labeling
verb=y
nqc=0
pixthresh=%d
thresh1=%f
thresh2=%f
thresh3=%f
"""%(pars.focptch,pars.defptch,pars.fltptch,pars.focprb,pars.defprb,deflbls,pars.begex,endex,# IO
     int(pars.pixthresh),float(pars.thresh1),float(pars.thresh2),float(pars.thresh3)) # Labeling
    # Write the par file
    with open(name,'w') as f:
      f.write(parout)

    return

