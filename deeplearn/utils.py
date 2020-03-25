"""
Deep learning utility functions.
Perform pre and post processing of training data
Also some plotting utlities

@author: Joseph Jennings
@version: 2020.03.13
"""
import sys
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import colors
from utils.image import remove_colorbar

def normalize(img,eps=sys.float_info.epsilon):
  """
  Normalizes an image accross channels  by removing
  the mean and dividing by the standard deviation
  """
  return (img - np.mean(img))/(np.std(img) + eps)

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
  length=img.shape[1]
  height=img.shape[0]
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
    for i in range(img.shape[0]):
      f = interpolate.interp2d(x,y,img[i,:,:],kind=kind)
      res[i,:,:] = f(xnew,ynew)
  elif len(img.shape)==2:
    f=interpolate.interp2d(x,y,img,kind=kind)
    res=f(xnew,ynew)

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
        kwargs.get("zmax",img.shape[0]),kwargs.get("zmin",0)],interpolation=kwargs.get("interp","none"))
  ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',14))
  ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',14))
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
        kwargs.get("zmax",img.shape[0]),kwargs.get("zmin",0)],interpolation=kwargs.get("interp","none"))
  ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',18))
  ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',18))
  ax.tick_params(labelsize=kwargs.get('ticksize',18))
  # Set colorbar
  cbar_ax = fig.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.12),
    kwargs.get('wbar',0.02),kwargs.get('hbar',0.75)])
  cbar = fig.colorbar(im,cbar_ax,format='%.1f',boundaries=np.arange(pmin,1.1,0.1))
  cbar.ax.tick_params(labelsize=kwargs.get('ticksize',18))
  cbar.set_label(kwargs.get('barlabel','Fault probablility'),fontsize=kwargs.get("barlabelsize",18))
  if(fname):
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
    plt.savefig(fname+"-prd.png",bbox_inches='tight',dpi=150,transparent=True)
    plt.close()
    # Crop and pad the image so they are the same size
    remove_colorbar(fname+"-img-tmp.png",cropsize=kwargs.get('cropsize',0),opath=fname+"-img.png")

