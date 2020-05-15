"""
Functions for creating labels for quantifying
image focusing

@author: Joseph Jennings
@version: 2020.05.12
"""
import numpy as np
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.utils import normalize
from resfoc.ssim import ssim
from scipy.signal.signaltools import correlate2d
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

