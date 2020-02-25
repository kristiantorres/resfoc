import numpy as np

def tpow(dat,nt,ot,dt,nx,tpow,nh=None,nro=None,norm=True):
  """ Applies a t^pow gain """
  if(nh != None and nro != None):
    return tpow4d(dat,nt,ot,dt,nx,tpow,nh,nro,norm)
  elif(nh == None and nro != None):
    return tpow3d(dat,nt,ot,dt,nx,tpow,nro,norm)
  else:
    return tpow2d(dat,nt,ot,dt,nx,tpow,norm)

def tpow2d(dat,nt,ot,dt,nx,tpow,norm=True):
  """ Applies a t^pow gain to the input array dat """
  # Build the t function
  if(ot != 0.0):
    t = np.linspace(ot,ot + (nt-1)*dt, nt)
  else:
    ot = dt
    t = np.linspace(ot,ot + (nt-1)*dt, nt)
  tp = np.power(t,tpow)
  # Normalize by default
  if(norm): tp = tp/np.max(tp)
  # Replicate it across the other axes
  tpx   = np.tile(np.array([tp]).T,(1,nx))
  # Scale the data
  return(dat*tpx).astype('float32')

def tpow3d(dat,nt,ot,dt,nx,tpow,nh,nro,norm=True):
  """ Applies a t^pow gain to the input array dat """
  # Build the t function
  if(ot != 0.0):
    t = np.linspace(ot,ot + (nt-1)*dt, nt)
  else:
    ot = dt
    t = np.linspace(ot,ot + (nt-1)*dt, nt)
  tp = np.power(t,tpow)
  # Normalize by default
  if(norm): tp = tp/np.max(tp)
  # Replicate it across the other axes
  tpx   = np.tile(np.array([tp]).T,(1,nx))
  tpxr  = np.tile(tpx.T,(nro,1,1))
  tpxrt = np.transpose(tpxr,(2,1,0))
  # Scale the data
  return (dat*tpxrt).astype('float32')

def tpow4d(dat,nt,ot,dt,nx,tpow,nh,nro,norm=True):
  """ Applies a t^pow gain to the input array dat """
  # Build the t function
  if(ot != 0.0):
    t = np.linspace(ot,ot + (nt-1)*dt, nt)
  else:
    ot = dt
    t = np.linspace(ot,ot + (nt-1)*dt, nt)
  tp = np.power(t,tpow)
  # Normalize by default
  if(norm): tp = tp/np.max(tp)
  # Replicate it across the other axes
  tpx   = np.tile(np.array([tp]).T,(1,nx))
  tpxh  = np.tile(tpx.T,(nh,1,1))
  tpxhr = np.tile(tpxh,(nro,1,1,1))
  tpxhrt = np.transpose(tpxhr,(3,2,1,0))
  # Scale the data
  return (dat*tpxhrt).astype('float32')

