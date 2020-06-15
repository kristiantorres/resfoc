"""
Functions for making predictions with keras models

@author: Joseph Jennings
@version: 2020.06.14
"""
import numpy as np
from deeplearn.utils import thresh, normalize, resizepow2
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from scaas.trismooth import smooth

def segmentfaults(img,mdl,nzp=128,nxp=128,strdz=None,strdx=None,verb=False):
  """
  Segments faults on a 2D image. Returns the probablility of each
  pixel being a fault or not.

  Parameters:
    img   - the input image [nz,nx]
    mdl   - the trained keras model
    nzp   - z-dimension of the patch provided to the CNN [128]
    nxp   - x-dimension of the patch provided to the CNN [128]
    strdz - z-dimension of the patch stride (50% overlap) [npz/2]
    strdx - x-dimension of the patch stride (50% overlap) [npx/2]
    verb  - verbosity flag [False]

  Returns the spatial fault probability map [nz,nx]
  """
  # Resample to nearest power of 2
  rimg = resizepow2(img,kind='linear')
  # Perform the patch extraction
  if(strdz is None): strdz = int(nzp/2)
  if(strdx is None): strdx = int(nxp/2)
  pe = PatchExtractor((nzp,nxp),stride=(strdx,strdz))
  iptch = pe.extract(rimg)
  numpz = iptch.shape[0]; numpx = iptch.shape[1]
  iptch = iptch.reshape([numpx*numpz,nzp,nxp,1])
  # Normalize each patch
  niptch = np.zeros(iptch.shape)
  for ip in range(numpz*numpx):
    niptch[ip,:,:] = normalize(iptch[ip,:,:])
  # Make a prediction
  iprd  = mdl.predict(niptch,verbose=verb)
  # Reconstruct the predictions
  ipra  = iprd.reshape([numpz,numpx,nzp,nxp])
  iprb  = pe.reconstruct(ipra)

  return iprb

def detectfaultpatch(img,mdl,nzp=64,nxp=64,strdz=None,strdx=None,rectx=30,rectz=30,verb=False):
  """
  Detects if a fault is present within an image or not

  Parameters:
    img   - the input image [nz,nx]
    mdl   - the trained keras model
    nzp   - z-dimension of the patch provided to the CNN [128]
    nxp   - x-dimension of the patch provided to the CNN [128]
    strdz - z-dimension of the patch stride (50% overlap) [npz/2]
    strdx - x-dimension of the patch stride (50% overlap) [npx/2]
    rectz - number of points to smooth in z direction [30]
    rectx - number of points to smooth in x direction [30]

  Returns a smooth probability map of detected faults
  """
  # Resample to nearest power of 2
  rimg = resizepow2(img,kind='linear')
  # Perform the patch extraction
  if(strdz is None): strdz = int(nzp/2)
  if(strdx is None): strdx = int(nxp/2)
  pe = PatchExtractor((nzp,nxp),stride=(strdx,strdz))
  iptch = pe.extract(rimg)
  numpz = iptch.shape[0]; numpx = iptch.shape[1]
  iptch = iptch.reshape([numpx*numpz,nzp,nxp,1])
  # Normalize and predict for each patch
  iprd = np.zeros(iptch.shape)
  for ip in range(numpz*numpx):
    iprd[ip,:,:] = mdl.predict(np.expand_dims(normalize(iptch[ip,:,:]),axis=0))
  # Reconstruct the predictions
  ipra  = iprd.reshape([numpz,numpx,nzp,nxp])
  iprb  = pe.reconstruct(ipra)

  # Smooth the predictions
  smprb = smooth(iprb.astype('float32'),rect1=rectx,rect2=rectz)

  return smprb

def focdefocflt(img,mdl,nzp=64,nxp=64,strdz=None,strdx=None,rectx=30,rectz=30,verb=False):
  """
  Classifies a fault as focused or defocused

  Parameters:
    img   - the input image
    mdl   - the trained keras model
    nzp   - z-dimension of the patch provided to the CNN [64]
    nxp   - x-dimension of the patch provided to the CNN [64]
    strdz - z-dimension of the patch stride (50% overlap) [npz/2]
    strdx - x-dimension of the patch stride (50% overlap) [npx/2]
    rectz - number of points to smooth in z direction [30]
    rectx - number of points to smooth in x direction [30]

  Returns a smooth probability map of focused/defocused faults
  """
  # Resample to nearest power of 2
  rimg = resizepow2(img,kind='linear')
  # Perform the patch extraction
  if(strdz is None): strdz = int(nzp/2)
  if(strdx is None): strdx = int(nxp/2)
  pe = PatchExtractor((nzp,nxp),stride=(strdx,strdz))
  iptch = pe.extract(rimg)
  numpz = iptch.shape[0]; numpx = iptch.shape[1]
  iptch = iptch.reshape([numpx*numpz,nzp,nxp,1])
  # Normalize and predict for each patch
  iprd = np.zeros(iptch.shape)
  for ip in range(numpz*numpx):
    iprd[ip,:,:] = mdl.predict(np.expand_dims(normalize(iptch[ip,:,:]),axis=0))
  # Reconstruct the predictions
  ipra  = iprd.reshape([numpz,numpx,nzp,nxp])
  iprb  = pe.reconstruct(ipra)

  # Smooth the predictions
  smprb = smooth(iprb.astype('float32'),rect1=rectx,rect2=rectz)

  return smprb

