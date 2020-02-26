import numpy as np
import resfoc.rstolt as rstolt
import resfoc.cosft as cft
import resfoc.depth2time as d2t
from deeplearn.utils import next_power_of_2

def pad_cft(n):
  np = next_power_of_2(n)
  if(np == n):
    return next_power_of_2(n+1) + 1 - n
  else:
    return np + 1 - n

def preresmig(img,ds,nro=6,oro=1.0,dro=0.01,time=True,verb=True,nthreads=4):
  # Get dimensions
  nh = img.shape[0]; nm = img.shape[1]; nz = img.shape[2]
  # Compute cosine transform
  nhp = pad_cft(nh); nmp = pad_cft(nm); nzp = pad_cft(nz)
  imgp   = np.pad(img,((0,nhp),(0,nmp),(0,nzp)),'constant')
  imgpft = cft.cosft(imgp,axis1=1,axis2=1,axis3=1).astype('float32')
  # Compute samplings
  dcs = cft.samplings(imgpft,ds)
  
  # Migration object
  nzpc = imgpft.shape[2]; nmpc = imgpft.shape[1]; nhpc = imgpft.shape[0]
  foro = oro - (nro-1)*dro; fnro = 2*nro-1
  if(verb): print("Rhos:",np.linspace(foro,foro + (fnro-1)*dro,2*nro-1))
  rst = rstolt.rstolt(nzpc,nmpc,nhpc,nro,dcs[2],dcs[1],dcs[0],dro,oro)
  
  ## Residual Stolt migration
  rmig = np.zeros([fnro,nhpc,nmpc,nzpc],dtype='float32')
  rst.resmig(imgpft,rmig,nthreads)
  
  # Inverse cosine transform
  rmigift = cft.icosft(rmig,axis2=1,axis3=1,axis4=1).astype('float32')
  rmigiftswind  = rmigift[:,0:nh,0:nm,0:nz]

  # Convert to time
  #if(time):
  #
  #  return rmigtme

  #else:

  return rmigiftswind

def rhoaxis(nro=6,oro=1.0,dro=0.01):
  return {'fnro':2*nro-1,'foro':oro - (nro-1)*dro,'fdro':dro}

def convert2time(nh,nm,nz,dz,nt,dt,depth,nro=6,oro=1.0,dro=0.01,vc=2000,oz=0.0,ot=0.0):
  # Compute axes
  foro = oro - (nro-1)*dro; fnro = 2*nro-1
  vel  = np.zeros(depth.shape,dtype='float32')
  time = np.zeros([fnro,nh,nm,nt],dtype='float32')
  # Apply a stretch for each rho
  for iro in range(fnro):
    ro = foro + iro*dro
    vel[:] = vc/ro
    print(vc/ro)
    d2t.convert2time(nh,nm,nz,0.0,dz,nt,0.0,dt,vel,depth[iro,:,:,:],time[iro,:,:,:])

  return time

