"""
Worker for patching fault segmentation data

@author: Joseph Jennings
@version: 2020.10.27
"""
import zmq, os, sys
import numpy as np
from comm.sendrecv import notify_server, send_zipped_pickle, recv_zipped_pickle
import subprocess
import inpout.seppy as seppy
from deeplearn.utils import normextract
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
  # Initialize the IO
  sep = seppy.sep()
  # Get the list of files
  imgs = chunk[0]['imgfiles']
  lbls = chunk[0]['lblfiles']
  # Output patched images
  oiptchs,olptchs = [],[]
  # Get the patching paramters
  ptchx = chunk[1]['nxp']
  ptchz = chunk[1]['nzp']
  for iex in progressbar(range(len(imgs)),"nex:",verb=chunk[3]['verb']):
    # Get the file names
    iimg,ilbl = imgs[iex],lbls[iex]
    if(chunk[3]['smooth']):
      oimg = os.path.splitext(iimg)[0] + '-sos.H'
    else:
      oimg = iimg
    if(not os.path.exists(oimg)):
      # Apply SOS to the file
      pyexe  = '/sep/joseph29/anaconda3/envs/py37/bin/python'
      sosexe = '/homes/sep/joseph29/projects/resfoc/bench/hale/scripts/SOSmoothing.py'
      sp = subprocess.check_call("%s %s -fin %s -fout %s"%(pyexe,sosexe,iimg,oimg),shell=True)
    # Read in the smoothed file
    saxes,smt = sep.read_file(oimg)
    smt = np.ascontiguousarray(smt.reshape(saxes.n,order='F'))
    # Read in the label
    laxes,lbl = sep.read_file(ilbl)
    lbl = np.ascontiguousarray(lbl.reshape(laxes.n,order='F'))
    # Patch the smoothed file and the label
    iptch = normextract(smt,nzp=ptchz,nxp=ptchx,norm=True)
    lptch = normextract(lbl,nzp=ptchz,nxp=ptchx,norm=False)
    oiptchs.append(iptch); olptchs.append(lptch)
  # Return the patched images and labels
  ochunk['imgs'] = np.concatenate(oiptchs,axis=0)
  ochunk['lbls'] = np.concatenate(olptchs,axis=0)
  # Return other parameters if desired
  ochunk['cid']  = chunk[2]
  # Tell server this is the result
  ochunk['msg'] = "result"
  # Send back the result
  send_zipped_pickle(socket,ochunk)
  # Receive 'thank you'
  socket.recv()

