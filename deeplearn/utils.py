"""
Deep learning utility functions.
Perform pre and post processing of training data
Also some plotting utlities

@author: Joseph Jennings
@version: 2002.02.08
"""
import sys
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import colors

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

def resample(img,new_shape,kind='linear'):
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
  #lr = new_shape[0]/length
  #hr = new_shape[1]/height
  #d1out = d1/hr
  #d2out = d2/lr
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

  return res
  #return res,[d1out,d2out]

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

def plotseglabel(img,lbl,show=False,color='red',**kwargs):
  """ Plots a binary label on top of an image """
  assert(img.shape == lbl.shape),'Input image and label must be same size'
  # Get mask
  mask = np.ma.masked_where(lbl == 0, lbl)
  # Select colormap
  cmap = colors.ListedColormap(['red','white'])
  fig = plt.figure(figsize=(kwargs.get('fsize1',8),kwargs.get('fsize2',6)))
  ax = fig.add_subplot(111)
  # Plot image
  ax.imshow(img,cmap=kwargs.get('cmap','gray'),
      vmin=kwargs.get('vmin',np.min(img)),vmax=kwargs.get('vmax',np.max(img)),
      extent=[kwargs.get("xmin",0),kwargs.get("xmax",img.shape[1]),
        kwargs.get("zmax",img.shape[0]),kwargs.get("zmin",0)])
  ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',18))
  ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',18))
  ax.tick_params(labelsize=kwargs.get('ticksize',18))
  ax.set_aspect(kwargs.get('aratio',1.0))
  # Plot label
  ax.imshow(mask,cmap=cmap)
  if(show):
    plt.show()

