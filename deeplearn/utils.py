"""
Deep learning utility functions.
Perform pre and post processing of training data
Also some plotting utlities

@author: Joseph Jennings
@version: 2020.05.28
"""
import sys
import numpy as np
from scipy import interpolate
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
import matplotlib.pyplot as plt
from matplotlib import colors
from genutils.image import remove_colorbar
from genutils.ptyprint import progressbar, create_inttag

def normalize(img,eps=sys.float_info.epsilon,mode='2d'):
  """
  Normalizes an image accross channels  by removing
  the mean and dividing by the standard deviation

  Parameters
    img - the input image. If the image has three dimensions,
          will normalize each image individually
    eps - parameters to avoid dividing a zero standard deviation
    mode - normalize in 2D or 3D (['2d'] or '3d')

  Returns normalized image(s)
  """
  if(mode == '2d'):
    if(len(img.shape) == 3):
      imgnrm = np.zeros(img.shape)
      nimg = img.shape[0]
      for k in range(nimg):
        imgnrm[k] = (img[k] - np.mean(img[k]))/(np.std(img[k]) + eps)
      return imgnrm
    else:
      return (img - np.mean(img))/(np.std(img) + eps)
  elif(mode == '3d'):
    if(len(img.shape) == 4):
      imgnrm = np.zeros(img.shape)
      nimg = img.shape[0]
      for k in range(nimg):
        imgnrm[k] = (img[k] - np.mean(img[k]))/(np.std(img[k]) + eps)
      return imgnrm
    else:
      return (img - np.mean(img))/(np.std(img) + eps)
  else:
    raise Exception("Mode not recognized")

def resizepow2(img,kind='linear'):
  """
  Resizes an image so that its dimensions are
  a power of 2
  """
  # Get input shape
  length=img.shape[-1]
  height=img.shape[-2]
  # Compute the new shape
  lengthnew = next_power_of_2(length)
  heightnew = next_power_of_2(height)
  new_shape = [heightnew, lengthnew]
  # Resample the image
  return resample(img,new_shape,kind)

def resample(img,new_shape,kind='linear',ds=[]):
  """
  Resamples an image. Can work up to 4D numpy arrays.
  assumes that the nz and nx axes are the last two (fastest)
  """
  # Original coordinates
  length=img.shape[-1]
  height=img.shape[-2]
  x=np.linspace(0,length,length)
  y=np.linspace(0,height,height)
  # New coordinates for interpolation
  xnew=np.linspace(0,length,new_shape[1])
  ynew=np.linspace(0,height,new_shape[0])
  # Compute new samplings
  if(len(ds) != 0):
      dout = []
      lr = new_shape[1]/length
      hr = new_shape[0]/height
      dout.append(ds[1]/lr)
      dout.append(ds[0]/hr)
  # Perform the interpolation
  if len(img.shape)==4:
    res = np.zeros([img.shape[0],img.shape[1],new_shape[0],new_shape[1]],dtype='float32')
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        f = interpolate.interp2d(x,y,img[i,j,:,:],kind=kind)
        res[i,j,:,:] = f(xnew,ynew)
  elif len(img.shape)==3:
    res = np.zeros([img.shape[0],new_shape[0],new_shape[1]],dtype='float32')
    for i in progressbar(range(img.shape[0]),"nimg:"):
      f = interpolate.interp2d(x,y,img[i,:,:],kind=kind)
      res[i,:,:] = f(xnew,ynew)
  elif len(img.shape)==2:
    res = np.zeros([new_shape[0],new_shape[1]],dtype='float32')
    f=interpolate.interp2d(x,y,img,kind=kind)
    res[:] = f(xnew,ynew)

  if(len(ds) == 0):
    return res
  else:
    return res,dout

def next_power_of_2(x):
  """ Gets the nearest power of two to x """
  return 1 if x == 0 else 2**(x - 1).bit_length()

def thresh(arr,thresh,mode='gt',absval=True):
  """ Applies a threshold to an array """
  out = np.zeros(arr.shape,dtype='float32')
  if(mode == 'eq'):
    idx = arr == thresh
    out[idx] = 1; out[~idx] = 0
  elif(mode == 'gt'):
    if(absval == True):
      idx = np.abs(arr) > thresh
    else:
      idx = arr > thresh
    out[idx] = 1; out[~idx] = 0
  elif(mode == 'ge'):
    if(absval == True):
      idx = np.abs(arr) >= thresh
    else:
      idx = arr >= thresh
    out[idx] = 1; out[~idx] = 0
  elif(mode == 'lt'):
    if(absval == True):
      idx = np.abs(arr) < thresh
    else:
      idx = arr < thresh
    out[idx] = 1; out[~idx] = 0
  elif(mode == 'le'):
    if(absval == True):
      idx = np.abs(arr) <= thresh
    else:
      idx = arr <= thresh
    out[idx] = 1; out[~idx] = 0

  return out

def plotseglabel(img,lbl,show=False,color='red',fname=None,**kwargs):
  """ Plots a binary label on top of an image """
  if(img.shape != lbl.shape):
    raise Exception('Input image and label must be same size')
  # Get mask
  mask = np.ma.masked_where(lbl == 0, lbl)
  # Select colormap
  cmap = colors.ListedColormap([color,'white'])
  fig = plt.figure(figsize=(kwargs.get('wbox',8),kwargs.get('hbox',6)))
  ax = fig.add_subplot(111)
  # Plot image
  ax.imshow(img,cmap=kwargs.get('cmap','gray'),
      vmin=kwargs.get('vmin',np.min(img)),vmax=kwargs.get('vmax',np.max(img)),
      extent=[kwargs.get("xmin",0),kwargs.get("xmax",img.shape[1]),
        kwargs.get("zmax",img.shape[0]),kwargs.get("zmin",0)],interpolation=kwargs.get("interp","sinc"))
  ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',14))
  ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',14))
  ax.set_title(kwargs.get('title',''),fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('ticksize',14))
  if(fname):
      ax.set_aspect(kwargs.get('aratio',1.0))
      plt.savefig(fname+"-img.png",bbox_inches='tight',dpi=150,transparent=True)
  # Plot label
  ax.imshow(mask,cmap=cmap,
      extent=[kwargs.get("xmin",0),kwargs.get("xmax",img.shape[1]),
        kwargs.get("zmax",img.shape[0]),kwargs.get("zmin",0)])
  ax.set_aspect(kwargs.get('aratio',1.0))
  if(show):
    plt.show()
  if(fname):
      plt.savefig(fname+"-lbl.png",bbox_inches='tight',dpi=150,transparent=True)
      plt.close()

def plotsegprobs(img,prd,pmin=0.01,alpha=0.5,show=False,fname=None,**kwargs):
  """ Plots unthresholded predictions on top of an image """
  if(img.shape != prd.shape):
    raise Exception('Input image and predictions must be same size')
  mask = np.ma.masked_where(prd <= pmin, prd)
  # Select colormap
  fig = plt.figure(figsize=(kwargs.get('wbox',8),kwargs.get('hbox',6)))
  ax = fig.add_subplot(111)
  # Plot image
  im = ax.imshow(img,cmap=kwargs.get('cmap','gray'),
      vmin=kwargs.get('vmin',np.min(img)),vmax=kwargs.get('vmax',np.max(img)),
      extent=[kwargs.get("xmin",0),kwargs.get("xmax",img.shape[1]),
        kwargs.get("zmax",img.shape[0]),kwargs.get("zmin",0)],interpolation=kwargs.get("interp","sinc"))
  ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',18))
  ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',18))
  ax.set_title(kwargs.get('title',''),fontsize=kwargs.get('labelsize',18))
  ax.tick_params(labelsize=kwargs.get('ticksize',18))
  # Set colorbar
  cbar_ax = fig.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.12),
    kwargs.get('wbar',0.02),kwargs.get('hbar',0.75)])
  cbar = fig.colorbar(im,cbar_ax,format='%.1f',boundaries=np.arange(pmin,1.1,0.1))
  cbar.ax.tick_params(labelsize=kwargs.get('ticksize',18))
  cbar.set_label(kwargs.get('barlabel','Fault probablility'),fontsize=kwargs.get("barlabelsize",18))
  if(fname):
    ftype = kwargs.get('ftype','png')
    ax.set_aspect(kwargs.get('aratio',1.0))
    plt.savefig(fname+"-img-tmp.png",bbox_inches='tight',dpi=150,transparent=True)
  cbar.remove()
  # Plot label
  imp = ax.imshow(mask,cmap='jet',
      extent=[kwargs.get("xmin",0),kwargs.get("xmax",img.shape[1]),
        kwargs.get("zmax",img.shape[0]),kwargs.get("zmin",0)],interpolation=kwargs.get("pinterp","bilinear"),
        vmin=pmin,vmax=1.0,alpha=alpha)
  ax.set_aspect(kwargs.get('aratio',1.0))
  # Set colorbar
  cbar_axp = fig.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.12),
    kwargs.get('wbar',0.02),kwargs.get('hbar',0.75)])
  cbarp = fig.colorbar(imp,cbar_axp,format='%.1f')
  cbarp.ax.tick_params(labelsize=kwargs.get('ticksize',18))
  cbarp.set_label(kwargs.get('barlabel','Fault probablility'),fontsize=kwargs.get("barlabelsize",18))
  cbarp.draw_all()
  if(show):
    plt.show()
  if(fname):
    plt.savefig(fname+"-prd."+ftype,bbox_inches='tight',dpi=150,transparent=True)
    plt.close()
    # Crop and pad the image so they are the same size
    remove_colorbar(fname+"-img-tmp.png",cropsize=kwargs.get('cropsize',0),oftype=ftype,opath=fname+"-img."+ftype)

def normextract(img,nzp=64,nxp=64,strdz=None,strdx=None,norm=True,flat=True):
  """
  Extract patches from an image and normalize each patch. Works for 2D
  and for 3D when the third dimension stride is one.

  Parameters:
    img   - the input image [n3,nz,nx]
    nzp   - size of the patch in z dimension [64]
    nxp   - size of the patch in x dimension [64]
    strdz - size of patch stride in z dimension [32]
    strdx - size of patch stride in x dimension [32]
    norm  - normalize the patches [True]
    flat  - return the patches flattened [nptch,nzp,nxp] or in a grid [numpz,numpx,nzp,nxp]

    Returns normalized image patches
  """
  if(strdz is None): strdz = int(nzp/2 + 0.5)
  if(strdx is None): strdx = int(nxp/2 + 0.5)
  if(len(img.shape) == 2):
    # Extract patches
    pe = PatchExtractor((nzp,nxp),stride=(strdz,strdx))
    ptch = pe.extract(img)

    # Get patch dimensions
    numpz = ptch.shape[0]; numpx = ptch.shape[1]

    # Flatten and normalize
    if(norm):
      ptchf = normalize(ptch.reshape([numpz*numpx,nzp,nxp]),mode='2d')
    else:
      ptchf = ptch.reshape([numpz*numpx,nzp,nxp])

  elif(len(img.shape) == 3):
    # Get size of third dimension
    n3 = img.shape[0]

    # Extract patches
    pea = PatchExtractor((n3,nzp,nxp),stride=(1,strdz,strdx))
    ptch = np.squeeze(pea.extract(img))

    # Get patch dimensions
    numpz = ptch.shape[0]; numpx = ptch.shape[1]

    # Flatten and normalize
    if(norm):
      ptchf = normalize(ptch.reshape([numpz*numpx,n3,nzp,nxp]),mode='3d')
    else:
      ptchf = ptch.reshape([numpz*numpx,n3,nzp,nxp])
  else:
    raise Exception("function supported only up to 3D")

  return ptchf

def torchprogress(cur :int,tot :int,loss :float,acc :float,size :int=40, file=sys.stdout) -> None:
  """
  Prints a progress bar during training of a torch
  neural network

  Parameters:
    cur  - index of the current batch
    tot  - the total number batches
    loss - the current running loss value
    acc  - the current accuracy

  Prints a progressbar to the screen
  """
  x = int(size*cur/tot)
  if(cur == 0): div = 1
  else: div = cur
  curform = create_inttag(cur,tot)
  file.write("%s/%d [%s%s] loss=%.4g acc=%.4f\r" % (curform,tot,"#"*x, "."*(size-x),loss/div,acc))
  if(cur == tot):
    file.write("\n")
  file.flush()

