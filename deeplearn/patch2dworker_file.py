"""
Worker for patching fault segmentation and focusing data

@author: Joseph Jennings
@version: 2020.09.20
"""
import zmq, os, sys
import numpy as np
from comm.sendrecv import notify_server, send_zipped_pickle, recv_zipped_pickle
import inpout.seppy as seppy
from deeplearn.dataloader import WriteToH5
import random, string
from resfoc.gain import agc
from deeplearn.utils import normextract
from genutils.ptyprint import progressbar

# Connect to socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://oas.stanford.edu:5555")

# Characters for output files
chars = string.ascii_uppercase + string.digits
tag   = '-' + ''.join(random.choice(chars) for _ in range(6))

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
  # Get the input file info
  fdict,faxes = sep.read_header(chunk[0]['ffile'],hdict=True)
  ddict,daxes = sep.read_header(chunk[0]['dfile'],hdict=True)
  rdict,raxes = sep.read_header(chunk[0]['rfile'],hdict=True)
  ldict,laxes = sep.read_header(chunk[0]['lfile'],hdict=True)
  [nz,na,naz,nx,nm] = faxes.n
  # Get the path to the binary
  fpath = os.path.splitext(fdict['in'])[0]; dpath = os.path.splitext(ddict['in'])[0]
  rpath = os.path.splitext(rdict['in'])[0]; lpath = os.path.splitext(ldict['in'])[0]
  # Segmentation files
  fsegname = fpath + '-seg' + tag + '.h5'
  wh5fseg  = WriteToH5(fsegname,dsize=1)
  dsegname = dpath + '-seg' + tag + '.h5'
  wh5dseg  = WriteToH5(dsegname,dsize=1)
  rsegname = rpath + '-seg' + tag + '.h5'
  wh5rseg  = WriteToH5(rsegname,dsize=1)
  wh5seg   = [wh5fseg,wh5dseg,wh5rseg]
  # Image focus files
  ffocname = fpath + '-foc' + tag + '.h5'
  wh5ffoc  = WriteToH5(ffocname,dsize=1)
  dfocname = dpath + '-foc' + tag + '.h5'
  wh5dfoc  = WriteToH5(dfocname,dsize=1)
  rfocname = rpath + '-foc' + tag + '.h5'
  wh5rfoc  = WriteToH5(rfocname,dsize=1)
  wh5foc   = [wh5ffoc,wh5dfoc,wh5rfoc]
  # Focus labels
  focval  = [1,-1,0]
  # Set the window parameters
  bxw = chunk[1]['bxw']
  exw = chunk[1]['exw']
  if(exw is None): exw = nx
  bzw = chunk[1]['bzw']
  ezw = chunk[1]['ezw']
  if(ezw is None): ezw = nz
  # Patching parameters
  nzp   = chunk[1]['nzp'];   nxp   = chunk[1]['nxp']
  strdz = chunk[1]['strdz']; strdx = chunk[1]['strdx']
  # File reading parameters
  nw = chunk[0]['nw']; nex = chunk[0]['nm']//nw
  k  = chunk[0]['fw']
  # Open a log file
  f = open(chunk[3]+'log'+tag+'.log','w')
  # Loop over all examples
  for iex in range(nex):
    f.write('Reading nw=%d from fw=%d...\n'%(nw,k)); f.flush()
    # Read in focused images
    faxes,foc = sep.read_wind(chunk[0]['ffile'],fw=k,nw=nw)
    foc   = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
    foct  = np.ascontiguousarray(np.transpose(foc[:,:,0,:,:],(0,2,1,3))) # [nw,nx,na,nz] -> [nw,na,nx,nz]
    # Read in defocused images
    daxes,dfc = sep.read_wind(chunk[0]['dfile'],fw=k,nw=nw)
    dfc   = np.ascontiguousarray(dfc.reshape(daxes.n,order='F').T).astype('float32')
    dfct  = np.ascontiguousarray(np.transpose(dfc[:,:,0,:,:],(0,2,1,3)))
    # Read in residually focused images
    raxes,rfc = sep.read_wind(chunk[0]['rfile'],fw=k,nw=nw)
    rfc   = np.ascontiguousarray(rfc.reshape(daxes.n,order='F').T).astype('float32')
    rfct  = np.ascontiguousarray(np.transpose(rfc[:,:,0,:,:],(0,2,1,3)))
    # Read in the labels
    laxes,lbl = sep.read_wind(chunk[0]['lfile'],fw=k,nw=nw)
    lbl = np.ascontiguousarray(lbl.reshape(laxes.n,order='F').T).astype('float32')
    f.write('Finished reading\n'); f.flush()
    # Window the data
    focw  = foct[:,:,bxw:exw,bzw:nz]
    dfcw  = dfct[:,:,bxw:exw,bzw:nz]
    rfcw  = rfct[:,:,bxw:exw,bzw:nz]
    lblw  = lbl [:,  bxw:exw,bzw:nz]
    fdrptch = [focw,dfcw,rfcw]
    lblptch = [lblw,lblw,np.zeros(lblw.shape,dtype='float32')]
    for ity in progressbar(range(len(fdrptch)),"nty:",file=f):
      ofoc = []; olbl = []; oseg = []
      for j in range(focw.shape[0]):
        # Transpose the images
        if(chunk[2]['transp']):
          datt = fdrptch[ity][j]
          lblt = lblptch[ity][j]
        else:
          datt = np.ascontiguousarray(np.transpose(fdrptch[ity][j],(0,2,1)))
          lblt = np.ascontiguousarray(lblptch[ity][j].T)
        # Extract patches
        dptch = normextract(datt,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=True)
        lptch = normextract(lblt,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=False)
        sptch = np.sum(dptch,axis=1)
        if(chunk[2]['agc']): gptch = agc(sptch.astype('float32'))
        else: gptch = sptch
        # Append to output lists
        ofoc.append(dptch); olbl.append(lptch); oseg.append(gptch)
      foc = np.concatenate(ofoc,axis=0)
      seg = np.concatenate(oseg,axis=0)
      lbl = np.concatenate(olbl,axis=0)

      # Write to files
      wh5seg[ity].write_examples(seg,lbl)

      foclbl = np.zeros([foc.shape[0],1],dtype='float32') + focval[ity]
      wh5foc[ity].write_examples(foc,foclbl)

    # Increment position in file
    k += nw

  # Delete WriteToH5 objects
  del wh5fseg; del wh5dseg; del wh5rseg; del wh5seg
  del wh5ffoc; del wh5dfoc; del wh5rfoc; del wh5foc

  # Output file parameters
  onames = {'fseg':fsegname,'dseg':dsegname,'rseg':rsegname,
            'ffoc':ffocname,'dfoc':dfocname,'rfoc':rfocname}
  ochunk['fnames'] = onames
  # Return other parameters if desired
  ochunk['cid']  = chunk[2]
  # Tell server this is the result
  ochunk['msg'] = "result"
  # Send back the result
  send_zipped_pickle(socket,ochunk)
  # Receive 'thank you'
  socket.recv()

