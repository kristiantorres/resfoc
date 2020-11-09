import numpy as np
import scaas.scaas2dpy as sca2d
from scaas.wavelet import ricker
import scaas.velocity as vel
from resfoc.tpow import tpow
import resfoc.cosft as cft
import resfoc.rstolt as rstolt
import inpout.seppy as seppy

def createdata_ptscat(nsx,osx,nz,nx,nh,nro,oro,dro,spacing=25,grid=False,keepoff=False,debug=False):
  """
  Creates a single training example: prestack residual migration images
  and the rho values to be estimated
  """
  ## Create the point scatterer model
  # Axes
  oz = 0.0; dz = 20.0
  ox = 0.0; dx = 20.0

  # Create modeling velocity
  modval  = 2500.0
  modvel  = np.zeros([nz,nx],dtype='float32') + modval
  romin = oro - (nro-1)*dro; romax = romin + dro*(2*nro-1)
  rhos = vel.create_randomptb(nz,nx,romin,romax,ncpu=8)
  modvel = modvel/rhos
  # Create migration velocity
  migval  = 2500.0
  migvel  = np.zeros([nz,nx],dtype='float32') + migval
  # Create perturbation
  dvel = None
  if(grid):
    dvel = vel.create_ptscatmodel(nz,nx,spacing,spacing)
  else:
    dvel = vel.create_randptscatmodel(nz,nx,100,spacing)
  if(debug):
    plt.imshow(dvel,cmap='gray',vmin=-1.0,vmax=1.0); plt.show()

  # Padding
  bx = 50; bz = 50
  modvelp  = np.pad(modvel, ((bz,bz),(bx,bx)),'edge')
  migvelp  = np.pad(migvel, ((bz,bz),(bx,bx)),'edge')
  dvelp    = np.pad(dvel,   ((bz,bz),(bx,bx)),'edge')

  # Pad for laplacian
  modvelp = np.pad(modvelp, ((5,5),(5,5)),'edge')
  migvelp = np.pad(migvelp, ((5,5),(5,5)),'edge')
  dvelp   = np.pad(dvelp,   ((5,5),(5,5)),'edge')

  # Get padded lengths
  nzp,nxp = modvelp.shape

  ## Model the born data
  osxp =  bx + 5 + osx; dsx = 10
  srczp = bz + 5
  nsrc = np.ones(nsx,dtype='int32')
  allsrcx = np.zeros([nsx,1],dtype='int32')
  allsrcz = np.zeros([nsx,1],dtype='int32')
  # All source x positions in one array
  srcs = np.linspace(osxp,osxp + (nsx-1)*dsx,nsx)
  for isx in range(nsx):
    allsrcx[isx,0] = int(srcs[isx])
    allsrcz[isx,0] = int(srczp)

  # Create receiver coordinates
  nrx = nx; orxp = bx + 5; drx = 1
  reczp = bz + 5
  nrec = np.zeros(nrx,dtype='int32') + nrx
  allrecx = np.zeros([nsx,nrx],dtype='int32')
  allrecz = np.zeros([nsx,nrx],dtype='int32')
  # Create all receiver positions
  recs = np.linspace(orxp,orxp + (nrx-1)*drx,nrx)
  for isx in range(nsx):
    allrecx[isx,:] = (recs[:]).astype('int32')
    allrecz[isx,:] = np.zeros(len(recs),dtype='int32') + reczp

  # Plot acquisition
  if(debug):
    plt.figure(1)
    # Plot velocity model
    vtot = modvel + dvel
    vtotp = modvelp + dvelp
    vmin = np.min(vtot); vmax = np.max(vtot)
    plt.imshow(vtotp,extent=[0,nxp,nzp,0],vmin=vmin,vmax=vmax,cmap='jet')
    # Get all source positions
    plt.scatter(allrecx[0,:],allrecz[0,:])
    plt.scatter(allsrcx[:,0],allsrcz[:,0])
    plt.show()

  # Create data axes
  ntu = 4000; otu = 0.0; dtu = 0.001;
  ntd = 1000; otd = 0.0; dtd = 0.004;

  # Create the wavelet array
  freq = 15; amp = 100.0; dly = 0.2;
  fsrc = ricker(ntu,dtu,freq,amp,dly)
  allsrcs = np.zeros([nsx,1,ntu],dtype='float32')
  for isx in range(nsx):
    allsrcs[isx,0,:] = fsrc[:]

  # Create output data array
  fact = int(dtd/dtu);
  ddat = np.zeros((nsx,ntd,nrx),dtype='float32')

  # Set up a wave propagation object
  sca = sca2d.scaas2d(ntd,nxp,nzp,dtd,dx,dz,dtu,bx,bz,alpha=0.99)

  # Forward modeling for all shots
  nthreads = 8
  sca.brnfwd(allsrcs,allsrcx,allsrcz,nsrc,allrecx,allrecz,nrec,nsx,modvelp,dvelp,ddat,nthreads)

  rnh = 2*nh + 1
  imgp  = np.zeros([rnh,nzp,nxp],dtype='float32')
  sca.brnoffadj(allsrcs,allsrcx,allsrcz,nsrc,allrecx,allrecz,nrec,nsx,migvelp,rnh,imgp,ddat,nthreads)

  # Output extended image
  imge = imgp[:,bz+5:nz+bz+5,bx+5:nx+bx+5]
  # Make z fast axis
  imget = np.ascontiguousarray(np.transpose(imge,(0,2,1)))
  iaxes = seppy.axes([rnh,nx,nz],[-dx*nh,0.0,0.0],[dx,dx,dz])

  ## Perform the residual migration of the prestack image
  # Pad before cosine transform
  imgep = np.pad(imget,((0,33-rnh),(0,513-nx),(0,513-nz)),'constant')

  # Cosine transform
  imgepft = cft.cosft(imgep,axis1=1,axis2=1,axis3=1).astype('float32')
  dcs = cft.samplings(imgepft,iaxes)

  # Migration object
  nzpc = imgepft.shape[2]; nmpc = imgepft.shape[1]; nhpc = imgepft.shape[0]
  foro = oro - (nro-1)*dro; fnro = 2*nro-1
  if(debug): print("Rhos:",np.linspace(foro,foro + (fnro-1)*dro,2*nro-1))
  rst = rstolt.rstolt(nzpc,nmpc,nhpc,nro,dcs[2],dcs[1],dcs[0],dro,oro)

  # Residual Stolt migration
  rmig = np.zeros([fnro,nhpc,nmpc,nzpc],dtype='float32')
  rst.resmig(imgepft,rmig,nthreads)

  # Inverse cosine transform
  rmigift = cft.icosft(rmig,axis2=1,axis3=1,axis4=1)

  rmigttp = None
  if(keepoff):
    # Remove the padding
    rmigiftswind  = rmigift[:,0:rnh,0:nx,0:nz]

    # Transpose
    rmigt = np.transpose(rmigiftswind,(3,2,1,0))
    rmigttp = tpow(rmigt,nz,0.0,dz,nx,1,rnh,fnro)
    if(debug):
      sep = seppy.sep([])
      sep.pltgreymovie(rmigttp[:,:,nh,:],greyargs="pclip=100 gainpanel=a",d3=dro,o3=foro,bg=True)
      sep.pltgreyimg(rhos,greyargs="newclip=1 color=j wantscalebar=y")

  else:
    # Remove the padding
    rmigiftswind  = rmigift[:,nh,0:nx,0:nz]

    # Transpose
    rmigt = np.transpose(rmigiftswind,(2,1,0))
    rmigttp = tpow(rmigt,nz,0.0,dz,nx,1,nh=None,nro=fnro)
    if(debug):
      sep = seppy.sep([])
      sep.pltgreymovie(rmigttp,greyargs="pclip=100 gainpanel=a",d3=dro,o3=foro,bg=True)
      sep.pltgreyimg(rhos,greyargs="newclip=1 color=j wantscalebar=y")

  return rmigttp,rhos
