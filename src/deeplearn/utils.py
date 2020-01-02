# Utility functions for processing data before 
# inputting to a neural network
import numpy as np
from scipy import interpolate

def resizepow2(img,kind='linear'):
  """ Resizes an image so that its dimensions are 
  a power of 2 """
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
  """ Resamples an image. Can work up to 4D numpy arrays.
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
    res = np.zeros([img.shape[0],img.shape[1],new_shape[0],new_shape[1]],dtype='float32')
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
