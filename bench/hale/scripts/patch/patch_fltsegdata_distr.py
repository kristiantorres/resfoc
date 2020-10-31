import inpout.seppy as seppy
import numpy as np
import glob
from client.sshworkers import create_host_list, launch_sshworkers, kill_sshworkers
from server.utils import startserver
from deeplearn.patch2dfltseg_chunkr import patch2dfltseg_chunkr
from server.distribute import dstr_collect
from deeplearn.utils import plot_seglabel
from deeplearn.dataloader import WriteToH5
from genutils.ptyprint import progressbar

# Start workers
#hosts = ['thing','storm','torch','fantastic','jarvis']
hosts = ['storm']
wph = len(hosts)*[1]
hin = create_host_list(hosts,wph)
cfile = "/homes/sep/joseph29/projects/resfoc/deeplearn/patch2dfltseg_worker.py"
launch_sshworkers(cfile,hosts=hin,sleep=1,verb=1,clean=False)

# Get files
ifiles = sorted(glob.glob('/homes/sep/joseph29/projects/resfoc/bench/hale/dat/split/img*.H'))
lfiles = sorted(glob.glob('/homes/sep/joseph29/projects/resfoc/bench/hale/dat/split/lbl*.H'))

ifilesf = []
# Remove all sos from files
for ifl in range(len(ifiles)):
  if('sos' not in ifiles[ifl]):
    ifilesf.append(ifiles[ifl])

nchnk = len(hin)
pcnkr = patch2dfltseg_chunkr(nchnk,ifilesf,lfiles,nzp=128,nxp=128,
                             smooth=False,verb=True)
gen = iter(pcnkr)

# Start the server
context,socket = startserver()

okeys = ['imgs','lbls']
output = dstr_collect(okeys,nchnk,gen,socket)

imgs = np.concatenate(output['imgs'],axis=0)
lbls = np.concatenate(output['lbls'],axis=0)

wh5seg = WriteToH5('/net/thing/scr2/joseph29/halefltseg_128nosm.h5',dsize=1)

wh5seg.write_examples(imgs,lbls)

# Clean up
kill_sshworkers(cfile,hosts,verb=False)

