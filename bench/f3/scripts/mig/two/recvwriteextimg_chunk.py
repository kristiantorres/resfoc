import inpout.seppy as seppy
import numpy as np
import time
from server.utils import startserver
from comm.sendrecv import recv_zipped_pickle, send_zipped_pickle
from comm.sendrecv import send_next_chunk
from client.sshworkers import launch_sshworkers, kill_sshworkers
from genutils.movie import viewcube3d

# Generator
def mygen(nchnks):
  ichnk = 0
  while(ichnk < nchnks):
    yield {'msg':"dummy"}
    ichnk += 1

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm','torch','thing','jarvis']
#hosts = ['fantastic','storm','torch']
cfile = "/homes/sep/joseph29/projects/resfoc/bench/f3/scripts/mig/two/readsendextimg_chunk.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1,clean=True)

# Bind to socket
context,socket = startserver()

img = np.zeros([41,100,500,500],dtype='float32')
nhx = img.shape[0]

n = len(hosts)
nouts = []

gen = mygen(n)

beg = time.time()
while(len(nouts)//nhx < n):
  rdict = recv_zipped_pickle(socket)
  if(rdict['msg'] == "available"):
    send_next_chunk(socket,gen,protocol=-1,zlevel=0)
  elif(rdict['msg'] == "result"):
    nouts.append(rdict['cid'])
    img[rdict['idx']] += rdict['img']
    socket.send(b"")

elapsed = (time.time() - beg)/60.0
print("Elapsed=%f"%(elapsed))

sep.write_file("f3extimgcopy.H",img.T/n,ds=[0.01,0.025,0.025,0.025],os=[0.0,0.0,0.0,0.0])

kill_sshworkers(cfile,hosts,verb=False)

