"""
Useful functions for plotting. No interactive plots.
See genutils.movie for interactive plotting
@author: Joseph Jennings
@version: 2020.08.20
"""
import numpy as np
from genutils.signal import ampspec1d
from resfoc.gain import agc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

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
    ax[0].set_title(kwargs.get('title',''),fontsize=kwargs.get('labelsize',14))
    maxval = np.max(wav)*1.5
    ax[0].set_ylim([-maxval,maxval])
    # Frequency domain
    spec,w = ampspec1d(wav,dt)
    ax[1].plot(w,spec/np.max(spec))
    ax[1].set_xlabel('Frequency (Hz)',fontsize=kwargs.get('labelsize',14))
    ax[1].tick_params(labelsize=kwargs.get('labelsize',14))
    ax[1].set_ylim([0,1.2])
    plt.subplots_adjust(hspace=kwargs.get('hspace',0.0))
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

def plot_allanggats(aimg,dz,dx,jx=10,transp=False,aagc=True,show=True,figname=None,**kwargs):
  """
  Makes a plot of all of the angle gathers by combining the spatial
  and angle axes

  Parameters
    aimg   - the angle domain image [nx,na,nz]
    transp - flag indicating that the input image has shape [na,nz,nx] [False]
    jx     - subsampling along the x axis (image points to skip) [10]
    aagc   - flag for applying agc [True]
  """
  if(transp):
    # [na,nz,nx] -> [nx,na,nz]
    aimgt = np.transpose(aimg,(2,0,1))
  else:
    aimgt = aimg
  nz = aimgt.shape[2]; na = aimgt.shape[1]; nx = aimgt.shape[0]
  # Subsample the spatial axis
  aimgts = aimgt[::jx,:,:]
  nxs = aimgts.shape[0]
  # Reshape to flatten the angle and CDP axes
  aimgtsnog = np.reshape(aimgts,[na*nxs,nz])
  if(aagc):
    aimgts = agc(aimgtsnog)
  else:
    aimgts = aimgtsnog
  # Min and max amplitudes
  vmin = kwargs.get('vmin',None); vmax = kwargs.get('vmax',None)
  if(vmin is None or vmax is None):
    vmin = np.min(aimgts); vmax = np.max(aimgts)
  # Plot the figure
  fig = plt.figure(figsize=(kwargs.get('wbox',10),kwargs.get('hbox',6)))
  ax = fig.gca()
  ax.imshow(aimgts.T,cmap='gray',extent=[kwargs.get('xmin',0.0),kwargs.get('xmax',nx*dx),kwargs.get('zmax',nz*dz),kwargs.get('zmin',0.0)],
      vmin=vmin*kwargs.get('pclip',1.0),vmax=vmax*kwargs.get('pclip',1.0),interpolation=kwargs.get('interp','sinc'))
  ax.set_xlabel('X (km)',fontsize=kwargs.get('labelsize',14))
  ax.set_ylabel('Z (km)',fontsize=kwargs.get('labelsize',14))
  ax.set_title(kwargs.get('title',' '),fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('labelsize',14))
  if(figname is None and show): plt.show()
  if(figname is not None):
    plt.savefig(figname,bbox_inches='tight',dpi=150,transparent=True)
    plt.close()

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
  lx = np.zeros(izmax-izmin) + (xloc+kwargs.get('ox',0.0))*dx
  vmin1 = kwargs.get('vmini',None); vmax1 = kwargs.get('vmaxi',None)
  if(vmin1 is None or vmax1 is None):
    vmin1 = np.min(mig); vmax1 = np.max(mig)
  ax1.imshow(mig[kwargs.get('xminwnd',0):kwargs.get('xmaxwnd',nx),kwargs.get('zminwnd',0):kwargs.get('zmaxwnd',nz)].T,cmap='gray',
             interpolation=kwargs.get('interp','sinc'),extent=[kwargs.get('xmin',0)*dx,
             kwargs.get('xmax',nx)*dx,izmax*dz,izmin*dz],vmin=vmin1*kwargs.get('pclip',1.0),vmax=vmax1*kwargs.get('pclip',1.0),
             aspect=kwargs.get('imgaspect',1.0))
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
  ax2.imshow(oneang[:,kwargs.get('zminwnd',0):kwargs.get('zmaxwnd',nz)].T,cmap='gray',
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

def plot_imgvelptb(img,velptb,dz,dx,thresh,aagc=True,alpha=0.3,show=False,figname=None,**kwargs):
  """
  Plots a velocity perturbation on top of an image

  Parameters
    focimg - the input image
    velptb - the velocity perturbation
    dz     - depth sampling interval
    dx     - horizontal sampling interval
    thresh - threshold in velocity to apply
    agc    - apply AGC to the image before plotting [True]
    alpha  - transparence value [0.3]
    xmin   - the minimum x sample to plot for windowing [0]
    xmax   - the maximum x sample to plot for windowing [nx]
    zmin   - the minimum z sample to plot for windowing [0]
    zmax   - the maximum z sample to plot for windowing [nz]
    pclip  - pclip to apply for gain                    [1.0]
  """
  if(img.shape != velptb.shape):
    raise Exception("Image and velocity must have same shape")
  # Get spatial plotting range
  [nz,nx] = img.shape;
  ixmin = kwargs.get('xmin',0); ixmax = kwargs.get('xmax',nx)
  xmin = ixmin*dx; xmax = ixmax*dx
  izmin = kwargs.get('zmin',0); izmax = kwargs.get('zmax',nz)
  zmin = izmin*dz; zmax = izmax*dz
  # Get amplitude range
  ivmin = kwargs.get('imin',np.min(img)); ivmax = kwargs.get('imax',np.max(img))
  pvmin = kwargs.get('velmin',np.min(velptb)); pvmax = kwargs.get('velmax',np.max(velptb))
  pclip = kwargs.get('pclip',1.0)
  # Plot the perturbation to get the true colorbar
  fig1 = plt.figure(figsize=(kwargs.get('wbox',10),kwargs.get('hbox',6)))
  ax1 = fig1.gca()
  im1 = ax1.imshow(velptb[izmin:izmax,ixmin:ixmax],cmap='seismic',
      extent=[ixmin*dx/1000,(ixmax-1)*dx/1000.0,izmax*dz/1000.0,izmin*dz/1000.0],interpolation='bilinear',
      vmin=pvmin,vmax=pvmax)
  plt.close()
  # Plot perturbation on the image
  fig = plt.figure(figsize=(kwargs.get('wbox',10),kwargs.get('hbox',6)))
  ax = fig.gca()
  if(aagc):
    gimg = agc(img.astype('float32').T).T
  else:
    gimg = img
  ax.imshow(gimg[izmin:izmax,ixmin:ixmax],vmin=ivmin*pclip,vmax=ivmax*pclip,
             extent=[ixmin*dx/1000,(ixmax)*dx/1000.0,izmax*dz/1000.0,izmin*dz/1000.0],cmap='gray',interpolation='sinc')
  mask1 = np.ma.masked_where((velptb) < thresh, velptb)
  mask2 = np.ma.masked_where((velptb) > -thresh, velptb)
  ax.imshow(mask1[izmin:izmax,ixmin:ixmax],extent=[ixmin*dx/1000,ixmax*dx/1000.0,izmax*dz/1000.0,izmin*dz/1000.0],alpha=alpha,
      cmap='seismic',vmin=pvmin,vmax=pvmax,interpolation='bilinear')
  ax.imshow(mask2[izmin:izmax,ixmin:ixmax],extent=[ixmin*dx/1000,ixmax*dx/1000.0,izmax*dz/1000.0,izmin*dz/1000.0],alpha=alpha,
      cmap='seismic',vmin=pvmin,vmax=pvmax,interpolation='bilinear')
  ax.set_xlabel('X (km)',fontsize=kwargs.get('labelsize',15))
  ax.set_ylabel('Z (km)',fontsize=kwargs.get('labelsize',15))
  ax.set_title(kwargs.get('title',''),fontsize=kwargs.get('labelsize',15))
  ax.tick_params(labelsize=kwargs.get('labelsize',15))
  # Colorbar
  cbar_ax = fig.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.15),kwargs.get('wbar',0.02),kwargs.get('hbar',0.70)])
  cbar = fig.colorbar(im1,cbar_ax,format='%.0f')
  cbar.ax.tick_params(labelsize=kwargs.get('labelsize',15))
  cbar.set_label('Velocity (m/s)',fontsize=kwargs.get('labelsize',15))
  if(figname is not None):
    plt.savefig(figname,bbox_inches='tight',transparent=True,dpi=150)
  if(show):
    plt.show()
  #plt.close()

def plot3d(data,os=[0.0,0.0,0.0],ds=[1.0,1.0,1.0],show=True,**kwargs):
  """
  Makes a 3D plot of a data cube

  Parameters:
    data - input 3D data cube
    os   - origins of each axis [0.0,0.0,0.0]
    ds   - samplings of each axis [1.0,1.0,1.0]
  """
  # Transpose if requested
  if(not kwargs.get('transp',False)):
    data = np.expand_dims(data,axis=0)
    data = np.transpose(data,(0,1,3,2))
  else:
    data = (np.expand_dims(data,axis=-1)).T
    data = np.transpose(data,(0,1,3,2))
  # Get the shape of the cube
  ns = np.flip(data.shape)
  # Make the coordinates for the cross hairs
  ds = np.append(np.flip(ds),1.0)
  os = np.append(np.flip(os),0.0)
  x1=np.linspace(os[0], os[0] + ds[0]*(ns[0]), ns[0])
  x2=np.linspace(os[1], os[1] + ds[1]*(ns[1]), ns[1])
  x3=np.linspace(os[2], os[2] + ds[2]*(ns[2]), ns[2])

  # Compute plotting min and max
  if(kwargs.get('vmin',None) == None or kwargs.get('vmax',None) == None):
    vmin = np.min(data)*kwargs.get('pclip',1.0)
    vmax = np.max(data)*kwargs.get('pclip',1.0)

  loc1 = kwargs.get('loc1',int(ns[0]/2*ds[0]+os[0]))
  i1 = int((loc1 - os[0])/ds[0])
  loc2 = kwargs.get('loc2',int(ns[1]/2*ds[1]+os[1]))
  i2 = int((loc2 - os[1])/ds[1])
  loc3 = kwargs.get('loc3',int(ns[2]/2*ds[2]+os[2]))
  i3 = int((loc3 - os[2])/ds[2])
  ax1 = None; ax2 = None; ax3 = None; ax4 = None
  curr_pos = 0

  # Axis labels
  label1 = kwargs.get('label1',' '); label2 = kwargs.get('label2',' '); label3 = kwargs.get('label3', ' ')

  width1 = kwargs.get('width1',4.0); width2 = kwargs.get('width2',4.0); width3 = kwargs.get('width3',4.0)
  widths=[width1,width3]
  heights=[width3,width2]
  gs_kw=dict(width_ratios=widths,height_ratios=heights)
  fig,ax=plt.subplots(2,2,figsize=(width1+width3,width2+width3),gridspec_kw=gs_kw)
  plt.subplots_adjust(wspace=0,hspace=0)

  title = kwargs.get('title',' ')
  ax[0,1].text(0.5,0.5,title[curr_pos],horizontalalignment='center',verticalalignment='center',fontsize=50)

  ## xz plane
  ax[1,0].imshow(data[curr_pos,:,i2,:],interpolation=kwargs.get('interp','none'),aspect='auto',
      extent=[os[0],os[0]+(ns[0])*ds[0],os[2]+ds[2]*(ns[2]),os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
  ax[1,0].tick_params(labelsize=kwargs.get('ticksize',14))
  ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
  ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')
  ax[1,0].set_xlabel(label1,fontsize=kwargs.get('labelsize',14))
  ax[1,0].set_ylabel(label3,fontsize=kwargs.get('labelsize',14))

  # yz plane
  im = ax[1,1].imshow(data[curr_pos,:,:,i1],interpolation=kwargs.get('interp','none'),aspect='auto',
      extent=[os[1],os[1]+(ns[1])*ds[1],os[2]+(ns[2])*ds[2],os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
  ax[1,1].tick_params(labelsize=kwargs.get('ticksize',14))
  ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
  ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
  ax[1,1].get_yaxis().set_visible(False)
  ax[1,1].set_xlabel(label2,fontsize=kwargs.get('labelsize',14))
  ax1=ax[1,1].twinx()
  ax1.set_ylim(ax[1,1].get_ylim())
  ax1.set_yticks([loc3])
  ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
  ax1.tick_params(labelsize=kwargs.get('ticksize',14))
  ax2=ax[1,1].twiny()
  ax2.set_xlim(ax[1,1].get_xlim())
  ax2.set_xticks([loc2])
  ax2.set_xticklabels(['%.2f'%(loc2)])
  ax2.tick_params(labelsize=kwargs.get('ticksize',14))

  # xy plane
  ax[0,0].imshow(np.flip(data[curr_pos,i3,:,:],0),interpolation=kwargs.get('interp','none'),aspect='auto',
      extent=[os[0],os[0]+(ns[0])*ds[0],os[1],os[1]+(ns[1])*ds[1]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
  ax[0,0].tick_params(labelsize=kwargs.get('ticksize',14))
  ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
  ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')
  ax[0,0].set_ylabel(label2,fontsize=kwargs.get('labelsize',14))
  ax[0,0].get_xaxis().set_visible(False)
  ax3=ax[0,0].twinx()
  ax3.set_ylim(ax[0,0].get_ylim())
  ax3.set_yticks([loc2])
  ax3.set_yticklabels(['%.2f'%(loc2)],rotation='vertical',va='center')
  ax3.tick_params(labelsize=kwargs.get('ticksize',14))
  ax4=ax[0,0].twiny()
  ax4.set_xlim(ax[0,0].get_xlim())
  ax4.set_xticks([loc1])
  ax4.set_xticklabels(['%.2f'%(loc1)])
  ax4.tick_params(labelsize=kwargs.get('ticksize',14))

  ax[0,1].axis('off')
  if(show):
    plt.show()

def plot_rhopicks(ang,smb,pck,dro,dz,oro,oz=0.0,doagc=False,mode='sbs',cnnpck=None,
                  show=True,figname=None,ftype='png',**kwargs):
  """
  Plots the semblance picks on top of the computed semblance panel
  and the residually migrated angle gathers

  Parameters:
    ang   - Residually migrated angle gathers [nro,na,nz]
    smb   - Computed rho semblance [nro,nz]
    pck   - The computed Rho picks [nz]
    dz    - The depth sampling
    dro   - The residual migration sampling
    oro   - The residual migration origin
    doagc - Apply agc to the gathers [False]
    mode  - Mode of how to plot ([sbs]/tb) side by side or top/bottom
    cnnpck - Plot CNN picks in addition to semblance picks
    show  - Show the plots [True]
    fname - Output figure name [None]
  """
  # Reshape the angle gathers
  nro = ang.shape[0]; na = ang.shape[1]; nz = ang.shape[2]
  angr = ang.reshape([na*nro,nz])
  # Gain the data
  if(doagc):
    angrg = agc(angr)
  else:
    angrg  = angr
  vmin = kwargs.get('vmin',np.min(angrg)); vmax = kwargs.get('vmax',np.max(angrg))
  pclip = kwargs.get('pclip',1.0)
  # Compute z for rho picks
  zmin = kwargs.get('zmin',oz); zmax = kwargs.get('zmax',oz+(nz-1)*dz)
  z = np.linspace(zmin,zmax,nz)
  # Plot the rho picks
  wbox = kwargs.get('wbox',14); hbox = kwargs.get('hbox',7)
  fntsize = kwargs.get('fontsize',15); tcksize = kwargs.get('ticksize',15)
  if(mode == 'sbs'):
    widthang = kwargs.get('widthang',2); widthrho = kwargs.get('widthrho',1)
    fig,ax = plt.subplots(1,2,figsize=(wbox,hbox),gridspec_kw={'width_ratios':[widthang,widthrho]})
    # Angle gather
    ax[0].imshow(angrg.T,cmap='gray',aspect=kwargs.get('angaspect',0.009),
                 extent=[oro,oro+(nro)*dro,kwargs.get('zmax',(nz-1)*dz),kwargs.get('zmin',0)],interpolation='sinc',
                 vmin=vmin*pclip,vmax=vmax*pclip)
    ax[0].plot(pck,z,linewidth=3,color='tab:cyan')
    if(cnnpck is not None):
      ax[0].plot(cnnpck,z,linewidth=3,color='tab:olive')
    ax[0].set_xlabel(r'$\rho$',fontsize=fntsize)
    ax[0].set_ylabel('Z (km)',fontsize=fntsize)
    ax[0].tick_params(labelsize=tcksize)
    # Semblance
    ax[1].imshow(smb.T,cmap='jet',aspect=kwargs.get('rhoaspect',0.02),
                 extent=[oro,oro+(nro)*dro,kwargs.get('zmax',nz*dz),kwargs.get('zmin',0.0)],interpolation='bilinear')
    ax[1].plot(pck,z,linewidth=3,color='k')
    if(cnnpck is not None):
      ax[1].plot(cnnpck,z,linewidth=3,color='gray')
    ax[1].set_xlabel(r'$\rho$',fontsize=fntsize)
    ax[1].set_ylabel(' ',fontsize=fntsize)
    ax[1].tick_params(labelsize=tcksize)
    plt.subplots_adjust(wspace=kwargs.get('wspace',-0.4))
  elif(mode == 'tb'):
    fig,ax = plt.subplots(2,1,figsize=(wbox,hbox))
    # Angle gather
    ax[0].imshow(angr.T,cmap='gray',aspect=0.009,extent=[oro,oro+(nro)*dro,nz*dz,0.0],interpolation='sinc',
               vmin=vmin*pclip,vmax=vmax*pclip)
    ax[0].plot(pck,z,linewidth=3,color='tab:cyan')
    ax[0].set_xlabel(r'$\rho$',fontsize=fntsize)
    ax[0].set_ylabel('Z (km)',fontsize=fntsize)
    ax[0].tick_params(labelsize=tcksize)
    # Semblance
    ax[1].imshow(smb.T,cmap='jet',aspect=0.009,extent=[oro,oro+(nro)*dro,nz*dz,0.0],interpolation='bilinear')
    ax[1].plot(pck,z,linewidth=3,color='k')
    ax[1].set_xlabel(r'$\rho$',fontsize=fntsize)
    ax[1].set_ylabel(' ',fontsize=fntsize)
    ax[1].tick_params(labelsize=tcksize)
  if(figname is not None):
    plt.savefig(figname+'.'+ftype,bbox_inches='tight',dpi=150,transparent=True)
    plt.close()
  if(show):
    plt.show()

def plot_cubeiso(data,os=[0.0,0.0,0.0],ds=[1.0,1.0,1.0],transp=False,show=True,figname=None,verb=True,**kwargs):
  """
  Makes an isometric plot of 3D data

  Parameters:
    data   - the input 3D data
    os     - the data origins [0.0,0.0.0]
    ds     - the data samplings [1.0,1.0,1.0]
    transp - flag for transposing second and third axes
    show   - flag for displaying the plot
    verb   - print the elevation and azimuth for the plot
  """
  if(transp):
    data = np.transpose(data,(0,2,1))
  # Get axes
  [n3,n2,n1] = data.shape
  [o3,o2,o1] = os
  [d3,d2,d1] = ds

  x1end = o1 + (n1-1)*d1
  x2end = o2 + (n2-1)*d2
  x3end = o3 + (n3-1)*d3

  # Build mesh grid for plotting
  x1 = np.linspace(o1, x1end, n1)
  x2 = np.linspace(o2, x2end, n2)
  x3 = np.linspace(o3, x3end, n3)
  x1ga, x3ga = np.meshgrid(x1,x3)
  x2ga, x3g  = np.meshgrid(x2, x3)
  x1g , x2gb = np.meshgrid(x1, x2)

  nlevels = kwargs.get('nlevels',200)

  # Get locations for extracting planes
  loc1 = kwargs.get('loc1',n1/2*d1+o1)
  i1 = int((loc1 - o1)/d1)
  loc2 = kwargs.get('loc2',n2/2*d2+o2)
  i2 = int((loc2 - o2)/d2)
  loc3 = kwargs.get('loc3',n3/2*d3+o3)
  i3 = int((loc3 - o3)/d3)

  # Get plotting range
  if(kwargs.get('stack',False)):
    # Get slices
    slc1 = data[:,:,i1]
    slc2 = data[:,i2,:]
    slc3 = np.sum(data,axis=0)
    # Set amplitude limits
    vmin1 = np.min(slc3); vmax1 = np.max(slc3)
    vmin2 = np.min(slc1); vmax2= np.max(slc1)
    levels1 = np.linspace(vmin1,vmax1,nlevels)
    levels2 = np.linspace(vmin2,vmax2,nlevels)
  else:
    # Get slices
    slc1 = data[:,:,i1]
    slc2 = data[:,i2,:]
    slc3 = data[i3,:,:]
    # Set amplitude limits
    vmin1 = kwargs.get('vmin',np.min(data))
    vmax1 = kwargs.get('vmax',np.max(data))
    vmin2 = vmin1; vmax2 = vmax1
    levels1 = np.linspace(vmin1,vmax1,nlevels)
    levels2 = np.linspace(vmin2,vmax2,nlevels)

  # Plot data
  fig = plt.figure(figsize=(kwargs.get('wbox',8),kwargs.get('hbox',8)))
  ax = fig.gca(projection='3d')

  cset = [[],[],[]]

  # Horizontal slice
  cset[0] = ax.contourf(x1ga, x3ga, slc2, zdir='z',offset=o2,levels=levels2,cmap='gray')

  # Into the screen slice
  cset[1] = ax.contourf(np.fliplr(slc1), x3g, np.flip(x2ga), zdir='x', offset=x1end,levels=levels2,cmap='gray')

  # Front slice
  cset[2] = ax.contourf(x1g, np.flipud(slc3), np.flip(x2gb), zdir='y', offset=o3,levels=levels1,cmap='gray')

  ax.set(xlim=[o1,x1end],ylim=[o3,x3end],zlim=[x2end,o2])

  fsize = kwargs.get('fsize',15)
  ax.set_xlabel(kwargs.get('x1label'),fontsize=fsize)
  ax.set_ylabel(kwargs.get('x2label'),fontsize=fsize)
  ax.set_zlabel(kwargs.get('x3label'),fontsize=fsize)
  ax.set_title(kwargs.get('title'),fontsize=fsize)
  ax.tick_params(labelsize=fsize)

  ax.view_init(elev=kwargs.get('elev',30),azim=kwargs.get('azim',-60))

  pts = [[(0.32,0.0),(0.32,0.64)]]
  pts2 = [[(0.0,0.32),(0.64,0.32)]]

  lines = LineCollection(pts,zorder=1000,color='k',lw=2)
  lines2 = LineCollection(pts2,zorder=1000,color='k',lw=2)
  #ax.add_collection3d(lines,zdir='y',zs=o2)
  #ax.add_collection3d(lines2,zdir='y',zs=o2)

  class FixZorderCollection(Line3DCollection):
    _zorder = 1000

    @property
    def zorder(self):
      return self._zorder

    @zorder.setter
    def zorder(self, value):
      pass

  if(verb):
    print("Elevation: %.3f Azimuth: %.3f"%(ax.elev,ax.azim))

  if(figname is None and show):
    plt.show()

  if(figname is not None):
    plt.savefig(figname,bbox_inches='tight',transparent=True,dpi=150)

