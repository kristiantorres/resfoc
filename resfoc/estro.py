"""
Functions for estimating the RMS velocity ratio (rho)
from residual migration images

@author: Joseph Jennings
@version: 2020.04.29
"""
import numpy as np
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from resfoc.ssim import ssim
from resfoc.rhoshifts import rhoshifts
from scipy.ndimage import map_coordinates

def estro_angs(resang,oro,dro,agc=True,rect1semb=10,rect2semb=3,smooth=True,rect1pick=40,rect2pick=40,gate=3,an=1,niter=100):
  """
  Computes semblance from residually migrated angle gathers
  and picks the rho from the maximum semblance

  Parameters
    resang    - input residually migrated angle gathers [nro,nx,na,nz]
    oro       - the origin of the rho axis
    dro       - the sampling of the rho axis
    agc       - apply AGC before computing semblance [True]
    rect1semb - number of points to smooth semblance on depth axis [10]
    rect2semb - number of points to smooth semblance on rho axis [3]
    smooth    - flag for smoothing the picked rho(z) function
    rect1pick - number of points to smooth the rho picks on depth axis [40]
    rect2pick - number of points to smooth the rho picks along the spatial axis [40]
    gate      - picking parameter (more to come) [3]
    an        - picking parameter (more to come) [1]
    niter     - number of iterations for shaping regularization smoothing [100]

  Returns an estimate of rho(x,z) with shape [nx,nz]
  """
  pass

def refocusimg(rimgs,rho,dro,ro1=None):
  """
  Refocuses the image based on an input rho map

  Parameters
    rimgs - input residually migrated images [nro,nx,nz].
            Input can be angle stack or zero offset.
    rho   - input smooth rho map
    dro   - the sampling along the rho axis
    ro1   - the index of the rho=1 image. [(nro-1)/2]

  Returns a refocused image [nx,nz]
  """
  # Get dimensions of the input images
  [nro,nx,nz] = rimgs.shape
  if(rho.shape[0] != nx or rho.shape[1] != nz):
    raise Exception("Input rho map must have same spatial dimensions as residually migrated images")

  # Get rho=1 index
  if(ro1 is None):
    ro1 = int((nro-1)/2)

  # Create coordinate array
  coords = np.zeros([3,nro,nx,nz],dtype='float32')

  # Compute the coordinates for shifting
  rhoshifts(nro,nx,nz,dro,rho,coords)

  # Apply shifts
  rfc = map_coordinates(rimgs,coords)

  # Extract at rho = 1
  return rfc[ro1]

def refocusang(resang,rho,dro):
  """
  Refocuses all angles based on an input rho map

  Parameters
    resang - input residually migrated angle gathers [nro,nx,na,nz]
    rho    - input smooth rho map
    dro    - the sampling along the rho axis
    ro1    - the index of the rho=1 image [(nro-1)/2]

    Returns a refocused image and flattened gathers [nx,na,nz]
  """
  pass

def estro_tgt(rimgs,fimg,dro,oro,nzp=128,nxp=128,strdx=64,strdz=64,transp=False,patches=False,onehot=False):
  """
  Estimates rho by comparing residual migration images with a
  well-focused "target image"

  Parameters
    rimgs   - input residual migration images [nro,nx,nz]
    fimg    - input target image [nx,nz]
    dro     - sampling along trial rho axis
    oro     - origin of rho axis
    nzp     - size of patches in z direction [128]
    nxp     - size of patches in x direction [128]
    strdx   - patch stride in x direction [128]
    strdz   - patch stride in z direction [128]
    transp  - transpose the input to have dimensions of [nro,nz,nx]
    patches - return the estimated rho in patch form  [numpx,numpz,nxp,nzp]
    onehot  - return the estimated rho in one-hot encoded form ((numpx,numpz,nro) with a one at the estimated rho)

  Returns the estimated rho in a map, patches and/or onehot form
  """
  if(transp):
    rimgst = np.transpose(rimgs,(0,2,1))
    fimgt  = fimg.T
  else:
    rimgst = rimgs
    fimgt  = fimg

  # Get dimensions
  [nro,nx,nz] = rimgst.shape

  if(nx != fimgt.shape[0] or nz != fimgt.shape[1]):
    raise Exception("Residual migration and focused image must have same spatial dimensions")

  # Extract patches on target image
  pe   = PatchExtractor((nxp,nzp),stride=(strdx,strdz))
  ptch = pe.extract(fimgt)
  numpx = ptch.shape[0]; numpz = ptch.shape[1]

  # Extract patches on residual migration image
  per = PatchExtractor((nro,nxp,nzp),stride=(nro,strdx,strdz))
  rptch = np.squeeze(per.extract(rimgst))

  # Allocate the output rho
  rhop = np.zeros(ptch.shape)
  oehs = np.zeros([numpx,numpz,nro])

  # Loop over each patch and compute the ssim for each rho
  for ixp in range(numpx):
    for izp in range(numpz):
      idx = ssim_ro(rptch[ixp,izp],ptch[ixp,izp])
      # Compute rho and save idx
      rhop[ixp,izp,:,:] = oro + idx*dro
      oehs[ixp,izp,idx] = 1.0

  # Reconstruct the rho field
  rhoi = pe.reconstruct(rhop)

  if(patches and onehot):
    return rhoi,rhop,oehs
  elif(patches):
    return rhoi,rhop
  elif(onehot):
    return rhoi,oehs
  else:
    return rhoi

def onehot2rho(oehs,dro,oro,nz=512,nx=1024,nzp=128,nxp=128,strdz=64,strdx=64,patches=False):
  """
  Builds a spatial rho map from onehot encoded vectors

  Parameters:
    oehs    - the input one-hot-encoded vectors [numpx,numpz,nro]
    dro     - the sampling along the rho axis
    oro     - the origin of the rho axis
    nz      - number of z samples of output rho image [512]
    nx      - number of x samples of output rho image [1024]
    nzp     - size of patch in z [128]
    nxp     - size of patch in x [128]
    strdx   - size of stride in x [64]
    strdz   - size of stride in z [64]
    patches - return rho as patches [False]

  Returns a spatial rho map
  """
  # Build patch extractor
  numpx = oehs.shape[0]; numpz = oehs.shape[1]
  pe = PatchExtractor((nxp,nzp),stride=(strdx,strdz))

  # Create output rhos
  rhoi = np.zeros([nx,nz])
  rhop = pe.extract(rhoi)

  # Loop over each patch
  for ixp in range(numpx):
    for izp in range(numpz):
      idx = np.argmax(oehs[ixp,izp])
      rhop[ixp,izp,:,:] = oro + idx*dro

  # Reconstruct the rho field
  rhoi = pe.reconstruct(rhop)

  if(patches):
    return rhoi,rhop
  else:
    return rhoi

def ssim_ro(rimgs,fimg):
  """
  Finds the rho that has the maximum SSIM for a specific target image

  Parameters:
    rimgs - Input residual migration images [nro,nx,nz]
    fimg  - Input well-focused image [nx,nz]
  """
  # Get dimensions
  nro = rimgs.shape[0]
  ssims = np.zeros(nro)
  for iro in range(nro):
    ssims[iro] = ssim(rimgs[iro],fimg)

  return np.argmax(ssims)

