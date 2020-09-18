"""
Worker for patching fault segmentation training data

@author: Joseph Jennings
@version: 2020.09.07
"""
import zmq
from comm.sendrecv import notify_server, send_zipped_pickle, recv_zipped_pickle
from resfoc.gain import agc
from deeplearn.utils import normextract
from genutils.ptyprint import progressbar
import numpy as np

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
  odat = []; olbl = []
  for iex in progressbar(range(nex),"nex:",verb=False):
    if(chunk[1]['agc']):
      datg = agc(dat[iex])
    else:
      datg = dat[iex]
    # Transpose the images
    if(chunk[1]['transp']):
      datgt = datg
      lblt  = lbl[iex]
    else:
      datgt = np.ascontiguousarray(datg.T)
      lblt  = np.ascontiguousarray(lbl[iex].T)
    # Extract patches
    dptch = normextract(datgt,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=True)
    lptch = normextract(lblt ,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=False)
    # Append to output lists
    odat.append(dptch); olbl.append(lptch)
  # Return patched data and labels
  ochunk['dat'] = np.concatenate(odat,axis=0)
  ochunk['lbl'] = np.concatenate(olbl,axis=0)
  # Return other parameters if desired
  ochunk['cid']  = chunk[2]
  # Tell server this is the result
  ochunk['msg'] = "result"
  # Send back the result
  send_zipped_pickle(socket,ochunk)
  # Receive 'thank you'
  socket.recv()

