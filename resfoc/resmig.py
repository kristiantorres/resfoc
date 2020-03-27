import numpy as np
import resfoc.rstolt as rstolt
import resfoc.cosft as cft
import resfoc.depth2time as d2t
from deeplearn.utils import next_power_of_2
from utils.ptyprint import printprogress

def pad_cft(n):
  """ Computes the size necessary to pad the image to next power of 2"""
  np = next_power_of_2(n)
  if(np == n):
    return next_power_of_2(n+1) + 1 - n
  else:
    return np + 1 - n

def preresmig(img,ds,nro=6,oro=1.0,dro=0.01,nps=None,time=True,transp=False,verb=True,nthreads=4):
  """
  Computes the prestack residual migration

  Parameters
    img      - the input prestack image. Axis nh,nx,nz
    ds       - the sampling of the image. [dh,dx,dz] (or [dh,dz,dx] if transp=True)
    nro      - the number of rhos for the residual migration [6]
    oro      - the center rho value [1.0]
    dro      - the spacing between the rhos [0.01]
    nps      - list of sizes that specify how much to pad for the cosine transform
               ([nhp,nxp,nzp] or [nhp,nzp,nxp] if transp=True)
    time     - return the output migration in time [True]
    transp   - take input [nh,nz,nx] and return output [nro,nh,nz,nx]
    verb     - verbosity flag [True]
    nthreads - number of CPU threads to use for computing the residual migration
  """
  if(transp):
    # [nh,nz,nx] -> [nh,nx,nz]
    iimg = np.ascontiguousarray(np.transpose(img,(0,2,1)))
    # [dh,dz,dx] -> [dh,dx,dz]
    ids = [ds[0],ds[2],ds[1]]
  else:
    iimg = img; ids = ds
  # Get dimensions
  nh = iimg.shape[0]; nm = iimg.shape[1]; nz = iimg.shape[2]
  if(nps is None):
    nhp = pad_cft(nh); nmp = pad_cft(nm); nzp = pad_cft(nz)
  else:
    if(transp):
      nhp = nps[0]; nmp = nps[2]; nzp = nps[1]
    else:
      nhp = nps[0]; nmp = nps[1]; nzp = nps[2]
  # Compute cosine transform
  imgp   = np.pad(iimg,((0,nhp),(0,nmp),(0,nzp)),'constant')
  if(verb): print("Padding to size nhp=%d nmp=%d nzp=%d"%(imgp.shape[0],imgp.shape[1],imgp.shape[2]))
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
  rst.resmig(imgpft,rmig,nthreads,verb)

  # Inverse cosine transform
  rmigift = cft.icosft(rmig,axis2=1,axis3=1,axis4=1,verb=True).astype('float32')
  rmigiftswind  = rmigift[:,0:nh,0:nm,0:nz]

  # Convert to time
  if(time):
    rmigtime = convert2time(rmigiftswind,ds[2],dt=0.004,oro=oro,dro=dro)
    if(transp):
      # [nro,nh,nx,nt] -> [nro,nh,nt,nx]
      return np.ascontiguousarray(np.transpose(rmigtime,(0,1,3,2)))
    else:
      # [nh,nx,nt]
      return rmigtime
  else:
    if(transp):
      # [nh,nx,nz] -> [nh,nz,nx]
      return np.ascontiguousarray(np.transpose(rmigiftswind,(0,1,3,2)))
    else:
      # [nh,nx,nz]
      return rmigiftswind

def get_rho_axis(nro=6,oro=1.0,dro=0.01):
  return 2*nro-1,oro - (nro-1)*dro,dro

def convert2time(depth,dz,dt,oro=1.0,dro=0.01,oz=0.0,ot=0.0,verb=False):
  """
  Converts residually migrated images from depth to time

  Parameters
    depth - the input depth residual depth migrated images
    dz    - the depth sampling of the residual migration images
    dt    - output time sampling
    oro   - center residual migration value [1.0]
    dro   - rho sampling [0.01]
    oz    - input depth origin [0.0]
    ot    - output time origin [0.0]
  """
  # Get the dimensions of the input cube
  fnro = depth.shape[0]; nh = depth.shape[1]; nm = depth.shape[2]; nz = depth.shape[3]
  nt = nz
  # Compute velocity
  T = (nt-1)*dt; Z = (nz-1)*dz
  vc = 2*Z/T
  # Compute rho axis
  nro = (fnro + 1)/2; foro = oro - (nro-1)*dro;
  vel  = np.zeros(depth.shape,dtype='float32')
  time = np.zeros([fnro,nh,nm,nt],dtype='float32')
  # Apply a stretch for each rho
  for iro in range(fnro):
    if(verb): printprogress("nrho:",iro,fnro)
    ro = foro + iro*dro
    vel[:] = vc/ro
    d2t.convert2time(nh,nm,nz,oz,dz,nt,ot,dt,vel,depth[iro,:,:,:],time[iro,:,:,:])
  if(verb): printprogress("nrho:",fnro,fnro)

  return time

