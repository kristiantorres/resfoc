"""
Useful functions for plotting. No interactive plots.
See utils.movie for interactive plotting
@author: Joseph Jennings
@version: 2020.03.24
"""
import numpy as np
from utils.signal import ampspec1d
from resfoc.gain import agc
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
    aimg - the angle domain image
    dx   - lateral sampling of the image
    dz   - depth sampling of the image
    oa   - origin of the angle axis
    da   - sampling of the angle axis
    xloc - the location at which to extract the angle gather [samples]
    show - flag of whether to display the image plot [True]
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
      cmap=kwargs.get('cmap','gray'),aspect=kwargs.get('aaspect',500))
  ax[1].set_xlabel(r'Angle ($\degree$)',fontsize=kwargs.get('labelsize',14))
  ax[1].set_ylabel(' ',fontsize=kwargs.get('labelsize',14))
  ax[1].tick_params(labelsize=kwargs.get('labelsize',14))
  ax[1].set_yticks([])
  plt.subplots_adjust(wspace=-0.4)
  if(show):
    plt.show()

def plot_allanggats(aimg,dz,dx,jx=10,transp=False,show=True,**kwargs):
  """
  Makes a plot of all of the angle gathers by combining the spatial
  and angle axes

  Parameters
    aimg   - the angle domain image [nx,na,nz]
    transp - flag indicating that the input image has shape [na,nz,nx] [False]
    jx     - subsampling along the x axis (image points to skip) [10]
  """
  if(transp):
    # [na,nz,nx] -> [nx,na,nz]
    aimgt = np.transpose(aimg,(2,0,1))
  else:
    aimgt = aimg
  nz = aimgt.shape[2]; na = aimgt.shape[1]; nx = aimg.shape[2]
  # Subsample the spatial axis
  aimgts = aimgt[::jx,:,:]
  nxs = aimgts.shape[0]
  # Reshape to flatten the angle and CDP axes
  aimgts = np.reshape(aimgts,[na*nxs,nz])
  # Min and max amplitudes
  vmin = np.min(aimgts); vmax = np.max(aimgts)
  # Plot the figure
  fig = plt.figure(figsize=(10,10))
  ax = fig.gca()
  ax.imshow(aimgts.T,cmap='gray',extent=[kwargs.get('xmin',0.0),nx*dx,nz*dz,kwargs.get('zmin',0.0)],
      vmin=vmin*kwargs.get('pclip',1.0),vmax=vmax*kwargs.get('pclip',1.0),interpolation=kwargs.get('interp','sinc'))
  ax.set_xlabel('X (km)',fontsize=kwargs.get('labelsize',14))
  ax.set_ylabel('Z (km)',fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('labelsize',14))
  if(show):
    plt.show()

def plot_anggatrhos(aimg,xloc,dz,dx,oro,dro,transp=False,figname=None,ftype='png',show=True,**kwargs):
  """
  Makes a plot of a single angle gather as it changes with rho

  Parameters
    aimg    - the residually migrated angle domain image [nro,nx,na,nz]
    xloc    - the image point at which to extract the angle gather [samples]
    dx      - the lateral sampling of the image
    dz      - the vertical sampling of the image
    oro     - the origin of the rho axis
    dro     - the sampling of the rho axis
    transp  - flag indicating whether to transpose the data [False]
    figname - name of output figure [None]
    ftype   - the type of output figure [png]
    show    - flag indicating whether to call plt.show() [True]
  """
  if(transp):
    # [nro,na,nz,nx] -> [nro,nx,na,nz]
    aimgt = np.transpose(aimg,(0,3,1,2))
  else:
    aimgt = aimg
  # Image dimensions
  nz = aimgt.shape[3]; na = aimgt.shape[2]; nx = aimgt.shape[1]; nro = aimgt.shape[0]

  ## Plot a line at the specified image point
  # Compute the original migrated image
  ro1dx = int((1-oro)/dro)
  if(kwargs.get('agc',False)):
    mig = agc(np.sum(aimgt[ro1dx],axis=1))
  else:
    mig = np.sum(aimgt[ro1dx],axis=1)
  fig1 = plt.figure(figsize=(kwargs.get('wboxi',14),kwargs.get('hboxi',7)))
  ax1 = fig1.gca()
  # Build the line
  izmin = kwargs.get('zmin',0); izmax = kwargs.get('zmax',nz)
  lz = np.linspace(izmin*dz,izmax*dz,izmax-izmin)
  lx = np.zeros(izmax-izmin) + xloc*dx
  vmin1 = np.min(mig); vmax1 = np.max(mig)
  ax1.imshow(mig[kwargs.get('xmin'):kwargs.get('xmax'),kwargs.get('zmin',0):kwargs.get('zmax',nz)].T,cmap='gray',
      interpolation=kwargs.get('interp','sinc'),extent=[kwargs.get('xmin',0)*dx,
    kwargs.get('xmax',nx)*dx,izmax*dz,izmin*dz],vmin=vmin1*kwargs.get('pclip',1.0),vmax=vmax1*kwargs.get('pclip',1.0))
  ax1.plot(lx,lz,color='k',linewidth=2)
  ax1.set_xlabel('X (km)',fontsize=kwargs.get('labelsize',14))
  ax1.set_ylabel('Z (km)',fontsize=kwargs.get('labelsize',14))
  ax1.tick_params(labelsize=kwargs.get('labelsize',14))
  if(figname is not None and show): plt.show()
  if(figname is not None):
    plt.savefig(figname+'-img.'+ftype,bbox_inches='tight',dpi=150,transparent=True)
    plt.close()

  ## Plot the rho figure
  # Grab a single angle gather
  if(kwargs.get('agc',False)):
    oneang = agc(aimgt[:,xloc,:,:])
  else:
    oneang = aimgt[:,xloc,:,:]
  # Flatten along angle and rho axis
  oneang = np.reshape(oneang,[nro*na,nz])
  # Min and max amplitudes
  vmin2 = np.min(oneang); vmax2 = np.max(oneang)
  fig2 = plt.figure(figsize=(kwargs.get('wboxg',14),kwargs.get('hboxg',7)))
  ax2 = fig2.gca()
  ax2.imshow(oneang[:,kwargs.get('zmin',0):kwargs.get('zmax',nz)].T,cmap='gray',
      extent=[oro,oro+nro*dro,kwargs.get('zmax',nz)*dz,kwargs.get('zmin',0.0)*dz],
      vmin=vmin2*kwargs.get('pclip',1.0),vmax=vmax2*kwargs.get('pclip',1.0),interpolation=kwargs.get('interp','sinc'),
      aspect=kwargs.get('roaspect',0.01))
  ax2.set_xlabel(r'$\rho$',fontsize=kwargs.get('labelsize',14))
  ax2.set_ylabel('Z (km)',fontsize=kwargs.get('labelsize',14))
  ax2.tick_params(labelsize=kwargs.get('labelsize',14))
  if(show): plt.show()
  if(figname is not None):
    plt.savefig(figname+'.'+ftype,bbox_inches='tight',dpi=150,transparent=True)
    plt.close()

