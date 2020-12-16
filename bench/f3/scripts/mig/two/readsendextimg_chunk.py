import zmq
import inpout.seppy as seppy
import numpy as np
from comm.sendrecv import notify_server, send_zipped_pickle, recv_zipped_pickle
from genutils.ptyprint import progressbar

# Connect to socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://oas.stanford.edu:5555")

# Listen for work from server
while True:
  # Notify we are ready
  notify_server(socket)
  # Get work
  chunk = recv_zipped_pickle(socket)
  # If chunk is empty, keep listening
  if(chunk == {}):
    continue
  # If I received something, do some work
  ochunk = {}
  # Read in the image
  sep = seppy.sep()
  iaxes,img = sep.read_file("/homes/sep/joseph29/projects/resfoc/bench/f3/f3imgextcritical.H")
  img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')
  nhx = img.shape[0]
  # Send back the result
  for ihx in progressbar(range(nhx),"transfer"):
    ochunk = {'msg':'result','cid':0,'idx':ihx}
    ochunk['img'] = img[ihx]
    send_zipped_pickle(socket, ochunk, protocol=-1, zlevel=0)
    # Receive 'thank you'
    socket.recv()

