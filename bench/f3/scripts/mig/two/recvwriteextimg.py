import inpout.seppy as seppy
import numpy as np
import time
from server.utils import startserver
from comm.sendrecv import recv_zipped_pickle, send_zipped_pickle
from client.sshworkers import launch_sshworkers, kill_sshworkers

# IO
sep = seppy.sep()

# Start workers
#hosts = ['fantastic','storm','torch','thing','jarvis']
hosts = ['fantastic']
cfile = "/homes/sep/joseph29/projects/resfoc/bench/f3/scripts/mig/two/readsendextimg.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1,clean=True)

# Bind to socket
context,socket = startserver()

beg = time.time()
ctr = 0
while(ctr < len(hosts)):
  rdict = recv_zipped_pickle(socket)
  if(rdict['msg'] == "available"):
    sdict = {}
    sdict['msg'] = 'start'
    send_zipped_pickle(socket,sdict,protocol=-1,zlevel=0)
  elif(rdict['msg'] == "result"):
    out = rdict['result']
    ctr += 1
    socket.send(b"")

elapsed = (time.time() - beg)/60.0
print("Elapsed=%f"%(elapsed))

