"""
Functions for creating labels for quantifying
image focusing

@author: Joseph Jennings
@version: 2020.05.12
"""
import numpy as np
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.utils import normalize, thresh
from resfoc.ssim import ssim
from scipy.signal.signaltools import correlate2d
from scaas.trismooth import smooth
import random
import matplotlib.pyplot as plt

def faultpatch_labels(img,fltlbl,nxp=64,nzp=64,strdx=32,strdz=32,pixthresh=20,norm=True,ptchimg=False,
                      qcptchgrd=False,dz=10,dx=10):
  """
  Assigns a zero or one to an image patch based on the number
  of fault pixels present within an image patch

  Parameters:
    img       - Input seismic image (to be patched) [nz,nx]
    fltlbl    - Input segmentation fault label [nz,nx]
    nxp       - Size of patch in x [64]
    nzp       - Size of patch in z [64]
    strdx     - Patch stride in x [32]
    strdz     - Patch stride in z [32]
    pixthresh - Number of fault pixels to determine if patch has fault
    ptchimg   - Return the reconstructed patch image [False]
    qcptchgrd - Makes a plot of the patch grid on the image
    dx        - Lateral sampling for plotting patch grid
    dz        - Vertical sampling for plotting patch grid

  Returns:
    Image and label patches [numpz,numpx,nzp,nxp] and the reconstructed
    label image
  """
  # Check that img and fltlbl are the same size
  if(img.shape[0] != fltlbl.shape[0] or img.shape[1] != fltlbl.shape[1]):
    raise Exception("Input image and fault label must have same dimensions")

  # Extract the patches
  pe = PatchExtractor((nzp,nxp),stride=(strdz,strdx))
  iptch = pe.extract(img)
  lptch = pe.extract(fltlbl)
  numpz = iptch.shape[0]; numpx = iptch.shape[1]

  if(qcptchgrd):
    nz = img.shape[0]; nx = img.shape[1]
    # Plot the patch grid
    bgz = 0; egz = (nz)*dz/1000.0; dgz = nzp*dz/1000.0
    bgx = 0; egx = (nx)*dx/1000.0; dgx = nxp*dx/1000.0
    zticks = np.arange(bgz,egz,dgz)
    xticks = np.arange(bgx,egx,dgx)
    fig = plt.figure(figsize=(10,6)); ax = fig.gca()
    ax.imshow(img,extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0],cmap='gray',interpolation='sinc')
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.grid(linestyle='-',color='k',linewidth=2)
    plt.show()

  # Output image patches
  iptcho = np.zeros(iptch.shape)

  # Output patch label
  ptchlbl = np.zeros(lptch.shape)

  # Check if patch contains faults
  for izp in range(numpz):
    for ixp in range(numpx):
      if(np.sum(lptch[izp,ixp]) >= pixthresh):
        ptchlbl[izp,ixp,:,:] = 1
      if(norm):
        iptcho[izp,ixp] = normalize(iptch[izp,ixp,:,:])
      else:
        iptcho[izp,ixp] = iptch[izp,ixp]

  # Reconstruct the patch label image
  ptchlblimg = pe.reconstruct(ptchlbl)

  if(ptchimg):
    return iptcho,ptchlbl,ptchlblimg
  else:
    return iptcho,ptchlbl

def focdefocflt_labels(dimg,fimg,fltlbl,nxp=64,nzp=64,strdx=32,strdz=32,pixthresh=20,metric='mse',focthresh=0.5,
                       norm=True,imgs=False,qcptchgrd=False,dz=10,dx=10):
  """
  Computes the fault-based focused and defocused labels

  Parameters
    dimg      - Input defocused image [nz,nx]
    fimg      - Input focused image [nz,nx]
    fltlbl    - Input fault labels [nz,nx]
    nxp       - Size of patch in x [64]
    nzp       - Size of patch in z [64]
    strdx     - Patch stride in x [32]
    strdz     - Patch stride in z [32]
    pixthresh - Number of fault pixels to determine if patch has fault [20]
    metric    - Metric for determining if fault is focused or not (['mse'] or 'ssim')
    focthresh - Threshold applied to metric to determining focusing [0.5]
    norm      - Normalize the images [True]
    imgs      - Return the label image and the norm image [False]
    qcptchgrd - Makes a plot of the patch grid on the image [False]
    dx        - Lateral sampling for plotting patch grid [10]
    dz        - Vertical sampling for plotting patch grid [10]
  """
  # Check that dimg, fimg and fltlbl are the same size
  if(dimg.shape[0] != fltlbl.shape[0] or dimg.shape[1] != fltlbl.shape[1]):
    raise Exception("Input image and fault label must have same dimensions")

  if(dimg.shape[0] != fimg.shape[0] or dimg.shape[1] != fimg.shape[1]):
    raise Exception("Input defocused image and defocused image must have same dimensions")

  # Patch extraction on the images
  pe = PatchExtractor((nzp,nxp),stride=(strdz,strdx))
  dptch = pe.extract(dimg)
  fptch = pe.extract(fimg)
  lptch = pe.extract(fltlbl)
  numpz = dptch.shape[0]; numpx = dptch.shape[1]

  if(qcptchgrd):
    nz = img.shape[0]; nx = img.shape[1]
    # Plot the patch grid
    bgz = 0; egz = (nz)*dz/1000.0; dgz = nzp*dz/1000.0
    bgx = 0; egx = (nx)*dx/1000.0; dgx = nxp*dx/1000.0
    zticks = np.arange(bgz,egz,dgz)
    xticks = np.arange(bgx,egx,dgx)
    fig = plt.figure(figsize=(10,6)); ax = fig.gca()
    ax.imshow(img,extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0],cmap='gray',interpolation='sinc')
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.grid(linestyle='-',color='k',linewidth=2)
    plt.show()

  # Output image patches
  dptcho = []; fptcho = []

  # Output patch label
  ptchlbl  = np.zeros(lptch.shape)
  lptcho = []

  # Norm image
  ptchnrm = np.zeros(lptch.shape)

  # Loop over each patch
  for izp in range(numpz):
    for ixp in range(numpx):
      # Check if patch contains faults
      if(np.sum(lptch[izp,ixp]) >= pixthresh):
        # Compute the desired norm between the two images
        if(metric == 'mse'):
          ptchnrm[izp,ixp,:,:] = mse(dptch[izp,ixp],fptch[izp,ixp])
          if(ptchnrm[izp,ixp,int(nzp/2),int(nxp/2)] >= focthresh):
            ptchlbl[izp,ixp,:,:] = 0
          else:
            ptchlbl[izp,ixp,:,:] = 1
        elif(metric == 'ssim'):
          ptchnrm[izp,ixp] = ssim(dptch[izp,ixp],fptch[izp,ixp])
          if(ptchnrm[izp,ixp,int(nzp/2),int(nxp/2)] >= focthresh):
            ptchlbl[izp,ixp,:,:] = 1
          else:
            ptchlbl[izp,ixp,:,:] = 0
        elif(metric == 'corr'):
          ndptch = normalize(dptch[izp,ixp]); nfptch = normalize(fptch[izp,ixp])
          #ptchnrm[izp,ixp] = np.max(correlate2d(ndptch,nfptch),mode='same'))
        else:
          raise Exception("Norm %s not yet implemented. Please try 'ssim' or 'mse'"%(metric))
        # Append label and image to output lists
        lptcho.append(ptchlbl[izp,ixp,int(nzp/2),int(nxp/2)])
        if(norm):
          dptcho.append(normalize(dptch[izp,ixp,:,:]))
          fptcho.append(normalize(fptch[izp,ixp,:,:]))
        else:
          dptcho.append(dptch[izp,ixp])
          fptcho.append(fptch[izp,ixp])

  # Convert to numpy arrays
  dptcho = np.asarray(dptcho)
  fptcho = np.asarray(fptcho)
  lptcho = np.asarray(lptcho)

  # Reconstruct the patch label image and patch norm image (for QC purposes)
  ptchlblimg = pe.reconstruct(ptchlbl)
  ptchnrmimg = pe.reconstruct(ptchnrm)

  if(imgs):
    return dptcho,fptcho,lptcho,ptchlblimg,ptchnrmimg
  else:
    return dptcho,fptcho,lptcho

def extract_defocpatches(dimg,fimg,fltlbl,nxp=64,nzp=64,strdx=32,strdz=32,pixthresh=20,metric='corr',focthresh=0.7,
                        norm=True,imgs=False,qcptchgrd=False,dz=10,dx=10):
  """
  Extracts defocused patches from a non-stationary
  defocused image

  Parameters
    dimg      - input defocused image [nz,nx]
    fimg      - input focused image [nz,nx]
    fltlbl    - input fault label [nz,nx]
    nxp       - size of patch in x direction [64]
    nzp       - size of patch in z direction [64]
    strdx     - size of stride in x direction [64]
    strdz     - size of stride in z direction [64]
    pixthresh - number of fault pixels required to keep a patch [20]
    metric    - metric for computing image similarity ['corr']
    focthresh - threshold for determing if image is focused [0.7]
    norm      - normalize patches before returning [True]
    qcptchgrd - plot patch grid for QC [False]
    dz        - z sampling for plotting patch grid [10]
    dx        - x sampling for plotting patch grid [10]

  Returns the selected defocused patches [nselect,nzp,nxp]
  """
  # Check that dimg, fimg and fltlbl are the same size
  if(dimg.shape[0] != fltlbl.shape[0] or dimg.shape[1] != fltlbl.shape[1]):
    raise Exception("Input image and fault label must have same dimensions")

  if(dimg.shape[0] != fimg.shape[0] or dimg.shape[1] != fimg.shape[1]):
    raise Exception("Input defocused image and defocused image must have same dimensions")

  # Patch extraction on the images
  pe = PatchExtractor((nzp,nxp),stride=(strdz,strdx))
  dptch = pe.extract(dimg)
  fptch = pe.extract(fimg)
  lptch = pe.extract(fltlbl)
  numpz = dptch.shape[0]; numpx = dptch.shape[1]

  if(qcptchgrd):
    nz = img.shape[0]; nx = img.shape[1]
    # Plot the patch grid
    bgz = 0; egz = (nz)*dz/1000.0; dgz = nzp*dz/1000.0
    bgx = 0; egx = (nx)*dx/1000.0; dgx = nxp*dx/1000.0
    zticks = np.arange(bgz,egz,dgz)
    xticks = np.arange(bgx,egx,dgx)
    fig = plt.figure(figsize=(10,6)); ax = fig.gca()
    ax.imshow(img,extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0],cmap='gray',interpolation='sinc')
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.grid(linestyle='-',color='k',linewidth=2)
    plt.show()

  # Output image patches
  dptcho = []; fptcho = []; sptcho = []

  # Norm image
  ptchnrm = np.zeros(lptch.shape)

  # Loop over each patch
  for izp in range(numpz):
    for ixp in range(numpx):
      # Check if patch contains faults
      if(np.sum(lptch[izp,ixp]) >= pixthresh):
        # Compute the desired norm between the two images
        if(metric == 'mse'):
          ptchnrm[izp,ixp,:,:] = mse(dptch[izp,ixp],fptch[izp,ixp])
          if(ptchnrm[izp,ixp,int(nzp/2),int(nxp/2)] > focthresh):
            if(norm):
              sptcho.append(normalize(dptch[izp,ixp]))
            else:
              sptcho.append(dptch[izp,ixp])
        elif(metric == 'ssim'):
          ptchnrm[izp,ixp] = ssim(dptch[izp,ixp],fptch[izp,ixp])
          if(ptchnrm[izp,ixp,int(nzp/2),int(nxp/2)] < focthresh):
            if(norm):
              sptcho.append(normalize(dptch[izp,ixp]))
            else:
              sptcho.append(dptch[izp,ixp])
        elif(metric == 'corr'):
          ptchnrm[izp,ixp] = corrsim(dptch[izp,ixp],fptch[izp,ixp])
          if(ptchnrm[izp,ixp,int(nzp/2),int(nxp/2)] < focthresh):
            if(norm):
              sptcho.append(normalize(dptch[izp,ixp]))
            else:
              sptcho.append(dptch[izp,ixp])
        else:
          raise Exception("Norm %s not yet implemented. Please try 'ssim','mse' or 'corr'"%(metric))
        if(norm):
          dptcho.append(normalize(dptch[izp,ixp,:,:]))
          fptcho.append(normalize(fptch[izp,ixp,:,:]))
        else:
          dptcho.append(dptch[izp,ixp])
          fptcho.append(fptch[izp,ixp])

  # Convert to numpy arrays
  dptcho = np.asarray(dptcho)
  fptcho = np.asarray(fptcho)
  sptcho = np.asarray(sptcho)

  # Reconstruct the patch norm image (for QC purposes)
  ptchnrmimg = pe.reconstruct(ptchnrm)

  if(imgs):
    return sptcho,dptcho,fptcho,ptchnrmimg
  else:
    return sptcho

def extract_fltptchs(fimg,fltlbl,nxp=64,nzp=64,strdx=32,strdz=32,pixthresh=20,
                         norm=True,qcptchgrd=False,dz=10,dx=10):
  """
  Extracts patches from a faulted image

  Parameters:
    fimg   - the input seismic image with faults [nz,nx]
    fltlbl - image containing the fault positions [nz,nx]
    nxp    - size of the patch in x [64]
    nzp    - size of the patch in z [64]
    strdx  - size of patch stride in x [32]
    strdz  - size of patch stride in z [32]
    pixthresh - number of fault pixels in image to determine if
                patch has a fault [20]
    norm   - normalize the patches [True]
    qcptchgrd - show a plot of the patch grid on the image [False]
    dz        - depth sampling for plotting patch grid [10]
    dx        - lateral sampling for plotting patch grid [10]

  Returns an array of image patches that contains faults [nptch,nzp,nxp]
  """
  # Check that dimg, fimg and fltlbl are the same size
  if(fimg.shape[0] != fltlbl.shape[0] or fimg.shape[1] != fltlbl.shape[1]):
    raise Exception("Input image and fault label must have same dimensions")

  # Patch extraction on the images
  pe = PatchExtractor((nzp,nxp),stride=(strdz,strdx))
  fptch = pe.extract(fimg)
  lptch = pe.extract(fltlbl)
  numpz = fptch.shape[0]; numpx = fptch.shape[1]

  # Output normalized patches
  nptch = []

  if(qcptchgrd):
    nz = fimg.shape[0]; nx = fimg.shape[1]
    # Plot the patch grid
    nz = fimg.shape[0]; nx = fimg.shape[1]
    # Plot the patch grid
    bgz = 0; egz = (nz)*dz/1000.0; dgz = nzp*dz/1000.0
    bgx = 0; egx = (nx)*dx/1000.0; dgx = nxp*dx/1000.0
    zticks = np.arange(bgz,egz,dgz)
    xticks = np.arange(bgx,egx,dgx)
    fig = plt.figure(figsize=(10,6)); ax = fig.gca()
    ax.imshow(fimg,extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0],cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
    ax.set_xlabel('X (km)',fontsize=15)
    ax.set_xlabel('Z (km)',fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.grid(linestyle='-',color='k',linewidth=2)
    plt.show()

  # Loop over each patch
  for izp in range(numpz):
    for ixp in range(numpx):
      # Check if patch contains faults
      if(np.sum(lptch[izp,ixp]) >= pixthresh):
        if(norm):
          nptch.append(normalize(fptch[izp,ixp]))
        else:
          nptch.append(fptch[izp,ixp])

  return np.asarray(nptch)

def flt_patches(iptch,lptch,pixthresh=20):
  """
  Returns a list of image patches that only contain faults.
  Similar to extract_focfltptchs but does not perform the
  patch extraction. Assumes input is in patch form

  Parameters:
    iptch     - image patches [numpz,numpx,nzp,nxp]
    lptch     - fault label patches [numpz,numpx,nzp,nxp]
    pixthresh - number of fault pixels in a patch to determine if it contains
                a fault [20]
  """
  # Check shapes
  if(iptch.shape != lptch.shape):
    raise Exception("Image patches and label patches must have same shape")

  # Get dimensions
  numpz = imgps.shape[0]; numpx = imgps.shape[1]

  # Output image fault patches
  fptch = []

  for izp in range(numpz):
    for ixp in range(numpx):
      # Check if patch contains faults
      if(np.sum(lptch[izp,ixp]) >= pixthresh):
        fptch.append(fptch[izp,ixp])

  return np.asarray(fptch)

def find_flt_patches(img,mdl,dz,mindepth,nzp=64,nxp=64,strdz=None,strdx=None,pthresh=0.2,nthresh=50,oz=0.0,
                     qcimgs=True):
  """
  Determines if patches contain a fault or not

  Parameters:
    img       - input fault seismic image [nz,nx]
    mdl       - fault segmentation keras CNN
    dz        - depth sampling
    mindepth  - minimum depth after which to look for faults
    nzp       - size of patch in x dimension [64]
    nxp       - size of patch in z dimension [64]
    strdz     - size of stride in z dimension [None]
    strdx     - size of stride in x dimension [None]
    pthresh   - probability threshold for determining if a pixel contains a fault [0.2]
    nthresh   - number of fault pixels in a patch to determined if it has a fault [50]
    oz        - depth origin [0.0]
    qcimgs    - flag for returning segmented fault image as well as fault patches
                for QC

  Returns a patch array where the patches are valued at either
  one (if patch contains a fault) or zero (if it does not have a fault)
  """
  # Get image dimensions
  nz = img.shape[0]; nx = img.shape[1]

  # Get strides
  if(strdz is None): strdz = int(nzp/2)
  if(strdx is None): strdx = int(nxp/2)

  # Extract patches on the image
  pe = PatchExtractor((nzp,nxp),stride=(strdz,strdx))
  iptch = pe.extract(img)
  # Flatten patches and make a prediction on each
  numpz = iptch.shape[0]; numpx = iptch.shape[1]
  iptchf = np.expand_dims(normalize(iptch.reshape([numpz*numpx,nzp,nxp])),axis=-1)
  fltpred = mdl.predict(iptchf)

  # Reshape the fault prediction array
  fltpred = fltpred.reshape([numpz,numpx,nzp,nxp])

  # Output arrays
  hasfault = np.zeros(iptch.shape)
  flttrsh  = np.zeros(iptch.shape)
  # Check if patch has a fault
  for izp in range(numpz):
    for ixp in range(numpx):
      # Compute current depth
      z = izp*strdz*dz + oz
      if(z > mindepth):
        # Threshold the patch
        flttrsh[izp,ixp] = thresh(fltpred[izp,ixp],pthresh)
        if(np.sum(flttrsh[izp,ixp]) > nthresh):
          hasfault[izp,ixp,:,:] = 1.0

  # Reconstruct the images for QC
  if(qcimgs):
    faultimg = pe.reconstruct(fltpred)
    thrshimg = pe.reconstruct(flttrsh)
    hsfltimg = pe.reconstruct(hasfault)

    return hasfault,hsfltimg,thresh(thrshimg,0.0),faultimg

  else:
    return hasfault

def semblance_power(img,transp=False):
  """
  A semblance metric for measuring flatness of angle gathers.

  Parameters:
    img - the input image [na,nx,nx]
  """
  if(len(img.shape) != 3):
    raise Exception("Input image must be 3D")

  stack   = np.sum(img,axis=0)
  stacksq = stack*stack
  num = smooth(stacksq.astype('float32'),rect1=3,rect2=10)

  sqstack = np.sum(img*img,axis=0)
  denom = smooth(sqstack.astype('float32'),rect1=3,rect2=10)

  semb = num/denom

  return np.sum(semb)

def mse(img,tgt):
  return np.linalg.norm(img-tgt)#/np.linalg.norm(img)

def corrsim(img,tgt):
  """
  A cross correlation image similarity metric

  Parameters:
    img - the input image
    tgt - the target image for comparison

  Returns a scalar metric for the similarity between images
  """
  # Normalize images
  normi = normalize(img); normt = normalize(tgt)
  # Cross correlation
  xcor = np.max(correlate2d(normi,normt,mode='same'))
  # Autocorrelations
  icor = np.max(correlate2d(normi,normi,mode='same'))
  tcor = np.max(correlate2d(normt,normt,mode='same'))
  # Similarity metric
  return xcor/np.sqrt(icor*tcor)

def varimax(img):
  """
  Computes the varimax norm on the input image. If the
  image is 2D, assumes that the input is the stack.
  If 3D, then sums over the middle axis

  Note that the order of the spatial axes does not matter

  Parameters:
    img - the input image (stacked or prestack) [nx,nz]/[nx,na,nz]

  Returns the varimax image entropy norm
  """
  if(len(img.shape) == 3):
    stk = np.sum(img,axis=1)
  else:
    stk = img
  nx = stk.shape[0]; nz = stk.shape[1]
  num = nz*nx*np.sum(stk**4)
  den = np.sum(stk**2)**2

  return num/den

