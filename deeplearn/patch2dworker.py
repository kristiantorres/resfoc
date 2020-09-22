"""
Worker for patching fault segmentation and focusing data

@author: Joseph Jennings
@version: 2020.09.17
"""
import zmq
from comm.sendrecv import notify_server, send_zipped_pickle, recv_zipped_pickle
from resfoc.gain import agc
from deeplearn.utils import normextract
from genutils.ptyprint import progressbar
import numpy as np
from socket import gethostname
import time, datetime

# Connect to socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://thing.stanford.edu:5555")

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
  # Get the size and data
  dat = chunk[0]['dat']; lbl = chunk[0]['lbl']
  nex = dat.shape[0]
  # Get parameters
  nzp   = chunk[0]['nzp'];   nxp   = chunk[0]['nxp']
  strdz = chunk[0]['strdz']; strdx = chunk[0]['strdx']
  # Output data and label
  ofoc = []; olbl = []; oseg = [];
  for iex in progressbar(range(nex),"nex:",verb=True):
    # Transpose the images
    if(chunk[1]['transp']):
      #TODO: First transpose and then agc, then transpose back
      datt = dat[iex]
      lblt = lbl[iex]
    else:
      datg = agc(dat[iex].astype('float32'))
      datt = np.ascontiguousarray(np.transpose(datg,(0,2,1)))
      lblt = np.ascontiguousarray(lbl[iex].T)
    # Extract patches
    dptch = normextract(datt,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=True)
    lptch = normextract(lblt ,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=False)
    sptch = np.sum(dptch,axis=1)
    if(chunk[1]['agc']): gptch = agc(sptch.astype('float32'))
    else: gptch = sptch
    # Append to output lists
    ofoc.append(dptch); olbl.append(lptch); oseg.append(gptch)
    #TODO: write outputs to a file like in file
  # Return patched data and labels
  ochunk['foc'] = 0#np.concatenate(ofoc,axis=0)
  ochunk['seg'] = 0#np.concatenate(oseg,axis=0)
  ochunk['lbl'] = 0#np.concatenate(olbl,axis=0)
  # Return other parameters if desired
  ochunk['cid']  = chunk[2]
  # Tell server this is the result
  ochunk['msg'] = "result"
  # Send back the result
  send_zipped_pickle(socket,ochunk,zlevel=0)
  # Receive 'thank you'
  socket.recv()

