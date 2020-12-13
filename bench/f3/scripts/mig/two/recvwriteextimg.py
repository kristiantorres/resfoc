import inpout.seppy as seppy
import numpy as np
import time
from server.utils import startserver
from comm.sendrecv import recv_zipped_pickle, send_next_chunk
from client.sshworkers import launch_sshworkers, kill_sshworkers

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm','torch','thing','jarvis']
cfile = "/homes/sep/joseph29/projects/resfoc/bench/f3/scripts/mig/readsendextimg.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1,clean=True)

# Bind to socket
context,socket = startserver()

done = False
rdict = recv_zipped_pickle(socket)
while(not done):
  if(rdict['msg'] == "available"):
    sdict = {}
    sdict['msg'] = 'start'
    send_zipped_pickle(socket,sdict,protocol=-1,zlevel=-1)
    beg = time.time()
  elif(rdict['msg'] == "result"):
    done = True
    out = rdict['result']
    elapsed = (time.time() - beg)/60.0
    print("Elapsed=%f"%(elapsed))


