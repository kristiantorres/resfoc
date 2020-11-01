import inpout.seppy as seppy
import numpy as np
import glob
from client.sshworkers import create_host_list, launch_sshworkers, kill_sshworkers
from server.utils import startserver
from deeplearn.soschunkr import soschunkr
from server.distribute import dstr_collect
from deeplearn.utils import plot_seglabel
from deeplearn.dataloader import WriteToH5
from genutils.ptyprint import progressbar

# Start workers
hosts = ['thing','storm','torch','fantastic','jarvis']
wph = len(hosts)*[5]
hin = create_host_list(hosts,wph)
cfile = "/homes/sep/joseph29/projects/resfoc/deeplearn/sosworker.py"
launch_sshworkers(cfile,hosts=hin,sleep=1,verb=1,clean=False)

# Get files
files  = glob.glob('/homes/sep/joseph29/projects/resfoc/bench/hale/dat/split_angs/f*.H')
files += glob.glob('/homes/sep/joseph29/projects/resfoc/bench/hale/dat/split_angs/r*.H')
files += glob.glob('/homes/sep/joseph29/projects/resfoc/bench/hale/dat/split_angs/d*.H')
files = sorted(files)

filesf = []
# Remove all sos from files
for ifl in range(len(files)):
  if('sos' not in files[ifl]):
    filesf.append(files[ifl])

nchnk = len(filesf)
scnkr = soschunkr(nchnk,filesf,verb=False)

gen = iter(scnkr)

# Start the server
context,socket = startserver()

okeys = ['imgs']
output = dstr_collect(okeys,nchnk,gen,socket)

# Clean up
kill_sshworkers(cfile,hosts,verb=False)

