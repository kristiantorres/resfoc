"""
Worker for patching fault segmentation data

@author: Joseph Jennings
@version: 2020.10.31
"""
import zmq, os, sys
import numpy as np
from comm.sendrecv import notify_server, send_zipped_pickle, recv_zipped_pickle
import subprocess
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
  # Get the list of files
  imgs = chunk[0]['imgfiles']
  # Output list of files
  ofiles = []
  for iex in progressbar(range(len(imgs)),"nex:",verb=chunk[-1]):
    # Get the file name
    iimg = imgs[iex]
    oimg = os.path.splitext(iimg)[0] + '-sos.H'
    if(not os.path.exists(oimg)):
      # Apply SOS to the file
      pyexe  = '/sep/joseph29/anaconda3/envs/py37/bin/python'
      sosexe = '/homes/sep/joseph29/projects/resfoc/bench/hale/scripts/SOSmoothing.py'
      sp = subprocess.check_call("%s %s -fin %s -fout %s"%(pyexe,sosexe,iimg,oimg),shell=True)
    ofiles.append(oimg)
  # Return the patched images and labels
  ochunk['imgs'] = ofiles
  # Return other parameters if desired
  ochunk['cid']  = chunk[2]
  # Tell server this is the result
  ochunk['msg'] = "result"
  # Send back the result
  send_zipped_pickle(socket,ochunk)
  # Receive 'thank you'
  socket.recv()

