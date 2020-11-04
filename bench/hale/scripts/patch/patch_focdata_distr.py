import numpy as np
from client.sshworkers import create_host_list, launch_sshworkers, kill_sshworkers
from server.utils import startserver
from deeplearn.patch2dfoc_chunkr import patch2dfoc_chunkr
from server.distribute import dstr_collect
from server.utils import splitnum
import random
from deeplearn.dataloader import WriteToH5
from genutils.ptyprint import progressbar

# Start workers
hosts = ['thing','storm','torch','fantastic','jarvis']
wph = len(hosts)*[2]
hin = create_host_list(hosts,wph)
cfile = "/homes/sep/joseph29/projects/resfoc/deeplearn/patch2dfoc_worker.py"
launch_sshworkers(cfile,hosts=hin,sleep=1,verb=1,clean=True)

# Generate list of numbers
nums = list(range(678))
idir = '/homes/sep/joseph29/projects/resfoc/bench/hale/dat/split_angs/'

wh5foc = WriteToH5('/net/thing/scr2/joseph29/halefoc_sm-small.h5',dsize=1)

# Start the server
context,socket = startserver()

fchcnks = splitnum(len(nums),32)
nchnk = 2
begpos = endpos = 0
for ichnk in range(nchnk):
  # Get the input chunk
  begpos = endpos; endpos += fchcnks[ichnk]
  inums = nums[begpos:endpos]
  # Create the chunkr
  nchnk  = len(inums)
  pcnkr  = patch2dfoc_chunkr(nchnk,inums,idir,nzp=64,nxp=64,pixthresh=75,
                             agc=True,smooth=True,verb=False)
  gen    = iter(pcnkr)
  okeys  = ['focs','defs','ress']
  output = dstr_collect(okeys,nchnk,gen,socket,verb=True)
  #TODO: for some reason the patches are not exactly 50% split
  #      need to investigate this
  # Process the output
  focs = np.concatenate(output['focs'],axis=0)
  defs = np.concatenate(output['defs'],axis=0)
  ress = np.concatenate(output['ress'],axis=0)
  nf = focs.shape[0]; nd = defs.shape[0]; nr = ress.shape[0]
  if(nd < nr):
    # Randomly select nrp from ress
    idxs1 = random.sample(range(nr), nr-nd)
    ressp = np.delete(ress,idxs1,axis=0)
    # Combine defocused
    adefs = np.concatenate([defs,ressp],axis=0)
    del ressp
  elif(nd > nr):
    alfa = nr/nd
    ndp  = int(alfa*nd)
    # Randomly select nrp from ress
    idxs1 = random.sample(range(nd), ndp)
    defsp = np.delete(defs,idxs1,axis=0)
    # Combine defocused
    adefs = np.concatenate([defsp,ress],axis=0)
    del defsp
  else:
    # Combine defocused
    adefs = np.concatenate([defs,ress],axis=0)
  del ress; del defs
  nu = len(adefs)
  if(nu > nf):
    idxs2 = random.sample(range(nu), nu-nf)
    adefs = np.delete(adefs,idxs2,axis=0)
  elif(nu < nf):
    idxs2 = random.sample(range(nf), nf-nu)
    focs = np.delete(focs,idxs2,axis=0)
  dlbls = np.zeros(adefs.shape[0])
  flbls = np.ones(focs.shape[0])

  # Write to the H5 file
  wh5foc.write_examples(focs ,flbls)
  wh5foc.write_examples(adefs,dlbls)

# Clean up
kill_sshworkers(cfile,hosts,verb=False)

