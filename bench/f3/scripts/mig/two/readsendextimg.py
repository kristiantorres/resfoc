import zmq
import inpout.seppy as seppy
from comm.sendrecv import notify_server, send_zipped_pickle, recv_zipped_pickle

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
  iaxes,img = sep.read_file("f3imgextcritical.H")
  img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')
  # Save it to the dictionary
  ochunk['result'] = img
  # Tell server this is the result
  ochunk['msg'] = "result"
  # Send back the result
  send_zipped_pickle(socket,ochunk)
  # Receive 'thank you'
  socket.recv()

