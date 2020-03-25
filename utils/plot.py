"""
Useful functions for plotting. No interactive plots.
See utils.movie for interactive plotting
@author: Joseph Jennings
@version: 2020.03.24
"""
import numpy as np
from utils.signal import ampspec1d
import matplotlib.pyplot as plt

def plot_wavelet(wav,dt,spectrum=True,show=True,**kwargs):
  """
  Makes a plot for a wavelet

  Parameters
    wav  - input wavelet
    dt   - sampling rate of wavelet
    show - flag whether to display the plot or not
  """
  t = np.linspace(0.0,(wav.shape[0]-1)*dt,wav.shape[0])
  if(spectrum):
    fig,ax = plt.subplots(2,1,figsize=(kwargs.get('wbox',15),kwargs.get('hbox',5)))
    # Time domain
    ax[0].plot(t,wav)
    ax[0].set_xlabel('Time (s)',fontsize=kwargs.get('labelsize',14))
    ax[0].tick_params(labelsize=kwargs.get('labelsize',14))
    maxval = np.max(wav)*1.5
    ax[0].set_ylim([-maxval,maxval])
    # Frequency domain
    spec,w = ampspec1d(wav,dt)
    ax[1].plot(w,spec/np.max(spec))
    ax[1].set_xlabel('Frequency (Hz)',fontsize=kwargs.get('labelsize',14))
    ax[1].tick_params(labelsize=kwargs.get('labelsize',14))
    ax[1].set_ylim([0,1.2])
    if(show):
      plt.show()
  else:
    # Only time domain
    fig = plt.figure(figsize=(kwargs.get('wbox',15),kwargs.get('hbox',3)))
    ax = fig.gca()
    ax.plot(t,wav)
    ax.set_xlabel('Time (s)',fontsize=kwargs.get('labelsize',14))
    ax.tick_params(labelsize=kwargs.get('labelsize',14))
    maxval = np.max(wav)*1.5
    plt.ylim([-maxval,maxval])
    if(show):
      plt.show()

def plot_imgpoff(oimg,dx,dz,zoff,xloc,oh,dh,show=True,**kwargs):
  """
  Makes a plot of the image and the extended axis at a specified location

  Parameters
    oimg - the extended image (either angles or subsurface offset)
    xloc - the location at which to extract the offset gather [samples]
  """
  # Get image dimensions
  nh = oimg.shape[0]; nz = oimg.shape[1]; nx = oimg.shape[2]
  fig,ax = plt.subplots(1,2,figsize=(kwargs.get('wbox',15),kwargs.get('hbox',8)),gridspec_kw={'width_ratios':[2,1]})
  # Plot the image
  ax[0].imshow(oimg[zoff,:,:],extent=[0.0,(nx)*dx,(nz)*dz,0.0],interpolation=kwargs.get('interp','sinc'),
    cmap=kwargs.get('cmap','gray'))
  # Plot a line at the specified image point
  lz = np.linspace(0.0,(nz)*dz,nz)
  lx = np.zeros(nz) + xloc*dx
  ax[0].plot(lx,lz,color='k',linewidth=2)
  ax[0].set_xlabel('X (km)',fontsize=kwargs.get('labelsize',14))
  ax[0].set_ylabel('Z (km)',fontsize=kwargs.get('labelsize',14))
  ax[0].tick_params(labelsize=kwargs.get('labelsize',14))
  # Plot the extended axis
  ax[1].imshow(oimg[:,:,xloc].T,extent=[oh,kwargs.get('hmax',oh+(nh+1)*dh),(nz)*dz,0.0],interpolation=kwargs.get('interp','sinc'),
      cmap=kwargs.get('cmap','gray'),aspect=1.0)
  ax[1].set_xlabel('Offset (km)',fontsize=kwargs.get('labelsize',14))
  ax[1].set_ylabel(' ',fontsize=kwargs.get('labelsize',14))
  ax[1].tick_params(labelsize=kwargs.get('labelsize',14))
  ax[1].set_yticks([])
  plt.subplots_adjust(wspace=-0.4)
  if(show):
    plt.show()

def plot_imgpang(aimg,dx,dz,xloc,oa,da,show=True,**kwargs):
  """
  Makes a plot of the image and the extended axis at a specified location

  Parameters
    aimg - the extend angle domain image
    xloc - the location at which to extract the angle gather [samples]
  """
  # Get image dimensions
  na = aimg.shape[0]; nz = aimg.shape[1]; nx = aimg.shape[2]
  fig,ax = plt.subplots(1,2,figsize=(kwargs.get('wbox',15),kwargs.get('hbox',8)),gridspec_kw={'width_ratios':[2,1]})
  # Plot the image
  ax[0].imshow(np.sum(aimg,axis=0),extent=[0.0,(nx)*dx,(nz)*dz,0.0],interpolation=kwargs.get('interp','sinc'),
    cmap=kwargs.get('cmap','gray'))
  # Plot a line at the specified image point
  lz = np.linspace(0.0,(nz)*dz,nz)
  lx = np.zeros(nz) + xloc*dx
  ax[0].plot(lx,lz,color='k',linewidth=2)
  ax[0].set_xlabel('X (km)',fontsize=kwargs.get('labelsize',14))
  ax[0].set_ylabel('Z (km)',fontsize=kwargs.get('labelsize',14))
  ax[0].tick_params(labelsize=kwargs.get('labelsize',14))
  # Plot the extended axis
  ax[1].imshow(aimg[:,:,xloc].T,extent=[oa,oa+(na)*da,(nz)*dz,0.0],interpolation=kwargs.get('interp','sinc'),
      cmap=kwargs.get('cmap','gray'),aspect=500)
  ax[1].set_xlabel(r'Angle ($\degree$)',fontsize=kwargs.get('labelsize',14))
  ax[1].set_ylabel(' ',fontsize=kwargs.get('labelsize',14))
  ax[1].tick_params(labelsize=kwargs.get('labelsize',14))
  ax[1].set_yticks([])
  plt.subplots_adjust(wspace=-0.4)
  if(show):
    plt.show()


