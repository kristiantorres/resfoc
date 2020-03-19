"""
Useful functions for creating movies for presentations
and viewing frames of a numpy array
@author: Joseph Jennings
@version: 2020.02.25
"""
import os
import inpout.seppy as seppy
from utils.ptyprint import create_inttag, progressbar
import numpy as np
import matplotlib.pyplot as plt

def makemovie_mpl(arr,odir,ftype='png',qc=False,skip=1,pttag=False,**kwargs):
  """ 
  Saves each frame on the fast axis to a png for viewing 

  Parameters:
    arr       - input array where the frames are on the fast axis
    odir      - the output directory where to save the frames
    ftype     - the type (extension) of figure to save [png]
    qc        - look at each frame as it is saved [False]
    skip      - whether to skip frames [1]
    pttag     - write the frame number in pretty (unix-friendly) format
    wbox      - the first dimension of the figure size (figure width) [10]
    hbox      - the second dimension of the figure size (figure height) [10]
    xmin      - the minimum lateral point in your data [0.0]
    xmax      - the maximum lateral point in your data (typically (nx-1)*dx) [nx]
    zmin      - the minimum depth point in your data [0.0]
    zmax      - the maximum depth point in your data (typically (nz-1)*dz) [nz]
    vmin      - the minumum value desired to be plotted (min(data))
    vmax      - the maximum value desired to be plotted (max(data))
    xlabel    - the label of the x-axis [None]
    ylabel    - the label of the y-axis [None]
    labelsize - the fontsize of the labels [18]
    ticksize  - the fontsize of the ticks [18]
    ttlstring - title to be printed. Can be printed of the form ttlstring%(ottl + dttl*(framenumber))
    ottl      - the first title value to be printed
    dttl      - the sampling of the title value to be printed
    aratio    - aspect ratio of the figure [1.0]
  """
  # Check if directory exists
  if(os.path.isdir(odir) == False):
    os.mkdir(odir)
  nfr = arr.shape[2]
  k = 0
  frames = np.arange(0,nfr,skip)
  for ifr in progressbar(frames, "frames"):
    fig = plt.figure(figsize=(kwargs.get("wbox",10),kwargs.get("hbox",10)))
    ax = fig.add_subplot(111)
    ax.imshow(arr[:,:,ifr],cmap=kwargs.get("cmap","gray"),
        extent=[kwargs.get("xmin",0),kwargs.get("xmax",arr.shape[1]),
          kwargs.get("zmax",arr.shape[0]),kwargs.get("zmin",0)],
        vmin=kwargs.get("vmin",np.min(arr[:,:,ifr])),
        vmax=kwargs.get("vmax",np.max(arr[:,:,ifr])))
    ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',18))
    ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',18))
    ax.tick_params(labelsize=kwargs.get('ticksize',18))
    if('%' in kwargs.get('ttlstring'),''):
      ax.set_title(kwargs.get('ttlstring','')%(kwargs.get('ottl',0.0) + kwargs.get('dttl',1.0)*k),
          fontsize=kwargs.get('labelsize',18))
    ax.set_aspect(kwargs.get('aratio',1.0))
    if(pttag):
      tag = create_inttag(k,nfr)
    else:
      tag = str(k)
    plt.savefig(odir+'/'+tag+'.'+ftype,bbox_inches='tight',dpi=kwargs.get("dpi",150))
    k += 1
    if(qc):
      plt.show()

def makemoviesbs_mpl(arr1,arr2,odir,ftype='png',qc=False,skip=1,pttag=False,**kwargs):
  """ 
  Saves each frame on the fast axis to a png for viewing 

  Parameters:
    arr1  - first input array (left panel) where the frames are on the fast axis
    arr2  - second input array (right panel) where the frames are on the fast axis
    odir  - the output directory where to save the frames
    ftype - the type (extension) of figure to save [png]
    qc    - look at each frame as it is saved [False]
    skip  - whether to skip frames [1]
    pttag - write the frame number in pretty (unix-friendly) format
  """
  # Check if directory exists
  if(os.path.isdir(odir) == False):
    os.mkdir(odir)
  assert(arr1.shape[2] == arr2.shape[2]),"Both movies must have same number of frames"
  nfr = arr1.shape[2]
  k = 0
  frames = np.arange(0,nfr,skip)
  for ifr in progressbar(frames, "frames"):
    # Setup plot
    f,ax = plt.subplots(1,2,figsize=(kwargs.get("figsize1",15),kwargs.get("figsize2",8)),
        gridspec_kw={'width_ratios': [kwargs.get("wratio1",1), kwargs.get("wratio2",1)]})
    # First plot
    im1 = ax[0].imshow(arr1[:,:,ifr],cmap=kwargs.get("cmap1","gray"),
        extent=[kwargs.get("xmin1",0),kwargs.get("xmax1",arr1.shape[1]),
          kwargs.get("zmax1",arr1.shape[0]),kwargs.get("zmin1",0)],
        vmin=kwargs.get("vmin1",np.min(arr1[:,:,ifr])),
        vmax=kwargs.get("vmax1",np.max(arr1[:,:,ifr])))
    ax[0].set_xlabel(kwargs.get('xlabel1',''),fontsize=kwargs.get('labelsize',18))
    ax[0].set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',18))
    ax[0].tick_params(labelsize=kwargs.get('ticksize',18))
    if('%' in kwargs.get('ttlstring1'),''):
      ax[0].set_title(kwargs.get('ttlstring1','')%(kwargs.get('ottl1',0.0) + kwargs.get('dttl1',1.0)*k),
          fontsize=kwargs.get('labelsize',18))
    else:
      ax[0].set_title(kwargs.get('ttlstring1',''),fontsize=kwargs.get('labelsize',18))
    # Second plot
    im2 = ax[1].imshow(arr2[:,:,ifr],cmap=kwargs.get("cmap2","gray"),
        extent=[kwargs.get("xmin2",0),kwargs.get("xmax2",arr2.shape[1]),
          kwargs.get("zmax2",arr2.shape[0]),kwargs.get("zmin2",0)],
        vmin=kwargs.get("vmin2",np.min(arr2[:,:,ifr])),
        vmax=kwargs.get("vmax2",np.max(arr2[:,:,ifr])))
    ax[1].set_xlabel(kwargs.get('xlabel2',''),fontsize=kwargs.get('labelsize',18))
    ax[1].tick_params(labelsize=kwargs.get('ticksize',18))
    ax[1].set_yticks([])
    if('%' in kwargs.get('ttlstring2','')):
      ax[1].set_title(kwargs.get('ttlstring2','')%(kwargs.get('ottl2',0.0) + kwargs.get('dttl2',1.0)*k),
          fontsize=kwargs.get('labelsize',18))
    else:
      ax[1].set_title(kwargs.get('ttlstring2',''),fontsize=kwargs.get('labelsize',18))
    # Color bar
    cbar_ax = f.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.12),
      kwargs.get('wbar',0.02),kwargs.get('hbar',0.75)])
    cbar = f.colorbar(im2,cbar_ax,format='%.2f')
    cbar.ax.tick_params(labelsize=kwargs.get('ticksize',18))
    cbar.set_label(kwargs.get('barlabel',''),fontsize=kwargs.get("barlabelsize",18))
    cbar.draw_all()
    # Spacing between plots
    plt.subplots_adjust(wspace=kwargs.get("pltspace",0.2))
    if(pttag):
      tag = create_inttag(k,nfr)
    else:
      tag = str(k)
    plt.savefig(odir+'/'+tag+'.'+ftype,bbox_inches='tight',dpi=kwargs.get("dpi",150))
    k += 1
    if(qc):
      plt.show()

def viewimgframeskey(data,transp=True,fast=True,show=True,**kwargs):
  """
  Provides a frame by frame interactive viewing of a 3D numpy array via the arrow keys.
  Assumes the slow axis is the first axis.

  Parameters:
    data      - an input 3D array. Must be 3D otherwise will fail
    transp    - whether to transpose the first and second axes before plotting
    cmap      - colormap [gray]
    vmin      - minimum value to display in the data [default is minimum amplitude of all data]
    vmax      - maximum value to display in the data [default is maximum amplitude of all data]
    pclip     - how much to clip the min and max of the amplitudes [0.9]
    ttlstring - title to be printed. Can be printed of the form ttlstring%(ottl + dttl*(framenumber))
    ottl      - origin for printing title values [0.0]
    dttl      - sampling for printing title values [1.0]
    scalebar  - flag of whether to plot a colorbar (False)
    hbar      - height of the scale bar
    wbar      - width of the scale bar
    zbar      - z position of scale bar
    xbar      - x position of scale bar
    barlabel  - colorbar label
    interp    - interpolation type for better display of the data (sinc for seismic, bilinear of velocity) [none]
    show      - flag for calling plt.show() [True]
  """
  if(len(data.shape) < 3):
    raise Exception("Data must be 3D")
  curr_pos = 0
  vmin = kwargs.get('vmin',None); vmax = kwargs.get('vmax',None)
  if(vmin == None or vmax == None):
    vmin = np.min(data)*kwargs.get('pclip',0.9)
    vmax = np.max(data)*kwargs.get('pclip',0.9)

  def key_event(e):
    nonlocal curr_pos,vmin,vmax

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % data.shape[0]

    if(transp):
      img = data[curr_pos,:,:].T
    else:
      img = data[curr_pos,:,:]
    if(not fast):
      ax.cla()
      ax.imshow(img,cmap=kwargs.get('cmap','gray'),vmin=vmin,vmax=vmax,
          extent=[kwargs.get('xmin',0.0),kwargs.get('xmax',data.shape[1]),
          kwargs.get('zmax',data.shape[2]),kwargs.get('zmin',0.0)],interpolation=kwargs.get('interp','none'))
    ax.set_title('%d'%(curr_pos),fontsize=kwargs.get('labelsize',14))
    ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',14))
    ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',14))
    #if(kwargs.get('scalebar',False)):
    #  cbar_ax = fig.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.12),
    #    kwargs.get('wbar',0.02),kwargs.get('hbar',0.75)])
    #  cbar = fig.colorbar(l,cbar_ax,format='%.2f')
    #  cbar.ax.tick_params(labelsize=kwargs.get('ticksize',18))
    #  cbar.set_label(kwargs.get('barlabel',''),fontsize=kwargs.get("barlabelsize",18))
    #  cbar.draw_all()
    if('%' in kwargs.get('ttlstring',' ')):
      ax.set_title(kwargs.get('ttlstring',' ')%(kwargs.get('ottl',0.0) + kwargs.get('dttl',1.0)*curr_pos),
          fontsize=kwargs.get('labelsize',14))
    else:
      ax.set_title(kwargs.get('ttlstring','%d'%curr_pos),fontsize=kwargs.get('labelsize',14))
    ax.tick_params(labelsize=kwargs.get('ticksize',14))
    if(fast):
      l.set_data(img)
    fig.canvas.draw()

  fig = plt.figure(figsize=(kwargs.get("wbox",10),kwargs.get("hbox",10)))
  fig.canvas.mpl_connect('key_press_event', key_event)
  ax = fig.add_subplot(111)
  # Show the first frame
  if(transp):
    img = data[0,:,:].T
  else:
    img = data[0,:,:]
  l = ax.imshow(img,cmap=kwargs.get('cmap','gray'),vmin=vmin,vmax=vmax,
      extent=[kwargs.get('xmin',0.0),kwargs.get('xmax',data.shape[1]),
        kwargs.get('zmax',data.shape[2]),kwargs.get('zmin',0.0)],interpolation=kwargs.get('interp','none'))
  ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',14))
  ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('ticksize',14))
  # Color bar
  if(kwargs.get('scalebar',False)):
    cbar_ax = fig.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.12),
      kwargs.get('wbar',0.02),kwargs.get('hbar',0.75)])
    cbar = fig.colorbar(l,cbar_ax,format='%.2f')
    cbar.ax.tick_params(labelsize=kwargs.get('ticksize',18))
    cbar.set_label(kwargs.get('barlabel',''),fontsize=kwargs.get("barlabelsize",18))
    cbar.draw_all()
  if('%' in kwargs.get('ttlstring',' ')):
    ax.set_title(kwargs.get('ttlstring',' ')%(kwargs.get('ottl',0.0)),fontsize=kwargs.get('labelsize',14))
  else:
    ax.set_title(kwargs.get('ttlstring','%d'%curr_pos),fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('ticksize',14))
  if(show):
    plt.show()

def viewpltframeskey(data,ox=0.0,dx=1.0,transp=True,show=True,**kwargs):
  """
  Provides a frame by frame interactive viewing of a 2D numpy array via the arrow keys.
  Assumes the slow axis is the first axis.

  Parameters:
    data      - an input 2D array. Must be 2D otherwise will fail
    vmin      - minimum value to display in the data [default is minimum amplitude of all data]
    vmax      - maximum value to display in the data [default is maximum amplitude of all data]
    pclip     - how much to clip the min and max of the amplitudes [0.9]
    ttlstring - title to be printed. Can be printed of the form ttlstring%(ottl + dttl*(framenumber))
    ottl      - origin for printing title values [0.0]
    dttl      - sampling for printing title values [1.0]
    show      - flag for calling plt.show() [True]
  """
  if(len(data.shape) < 2):
    raise Exception("Data must be 2D")
  curr_pos = 0
  # Create the x axis
  nx = data.shape[1]
  xs = np.linspace(ox,ox+(nx-1)*dx,nx)
  # Find the min and the max of the frames
  if(kwargs.get('ymin',None) == None or kwargs.get('ymax',None) == None):
    ymin = np.min(data); ymax = np.max(data)

  def key_event(e):
    nonlocal curr_pos,xs

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % data.shape[0]

    if(transp):
      crv = data[curr_pos,:].T
    else:
      crv = data[curr_pos,:]
    ax.cla()
    ax.plot(xs,crv,color=kwargs.get('color','tab:blue'),linewidth=kwargs.get('linewidth',1.5))
    ax.set_ylim([kwargs.get('ymin',ymin),kwargs.get('ymax',ymax)])
    ax.set_title('%d'%(curr_pos),fontsize=kwargs.get('labelsize',14))
    ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',14))
    ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',14))
    if('%' in kwargs.get('ttlstring',' ')):
      ax.set_title(kwargs.get('ttlstring',' ')%(kwargs.get('ottl',0.0) + kwargs.get('dttl',1.0)*curr_pos),
          fontsize=kwargs.get('labelsize',14))
    else:
      ax.set_title(kwargs.get('ttlstring','%d'%curr_pos),fontsize=kwargs.get('labelsize',14))
    ax.tick_params(labelsize=kwargs.get('ticksize',14))
    fig.canvas.draw()

  fig = plt.figure(figsize=(kwargs.get("wbox",10),kwargs.get("hbox",10)))
  fig.canvas.mpl_connect('key_press_event', key_event)
  ax = fig.add_subplot(111)
  # Show the first frame
  if(transp):
    crv = data[0,:].T
  else:
    crv = data[0,:,:]
  l = ax.plot(xs,crv,color=kwargs.get('color','tab:blue'),linewidth=kwargs.get('linewidth',1.5))
  ax.set_ylim([kwargs.get('ymin',ymin),kwargs.get('ymax',ymax)])
  ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',14))
  ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('ticksize',14))
  if('%' in kwargs.get('ttlstring',' ')):
    ax.set_title(kwargs.get('ttlstring',' ')%(kwargs.get('ottl',0.0)),fontsize=kwargs.get('labelsize',14))
  else:
    ax.set_title(kwargs.get('ttlstring','%d'%curr_pos),fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('ticksize',14))
  if(show):
    plt.show()

def viewframessld(data,transp=True,**kwargs):
  pass

def viewcube3d(data,os=[0.0,0.0,0.0],ds=[1.0,1.0,1.0],**kwargs):
  """
  Plots three frames of a 3D plot for visualization. Allows
  for user interaction with the arrow keys or e,w,n,s keys.
  Originally written by Huy Le with some of my modifications

  Parameters:
    data - input data cube (numpy array)
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
  x1=np.linspace(os[0], os[0] + ds[0]*(ns[0]-1), ns[0])
  x2=np.linspace(os[1], os[1] + ds[1]*(ns[1]-1), ns[1])
  x3=np.linspace(os[2], os[2] + ds[2]*(ns[2]-1), ns[2])

  # Compute plotting min and max
  if(kwargs.get('vmin',None) == None or kwargs.get('vmax',None) == None):
    vmin = np.min(data)*kwargs.get('pclip',0.9)
    vmax = np.max(data)*kwargs.get('pclip',0.9)

  # Define nonlocal variables
  loc1 = kwargs.get('loc1',int(ns[0]/2*ds[0]+os[0]))
  i1 = int((loc1 - os[0])/ds[0])
  loc2 = kwargs.get('loc2',int(ns[1]/2*ds[1]+os[1]))
  i2 = int((loc2 - os[1])/ds[1])
  loc3 = kwargs.get('loc3',int(ns[2]/2*ds[2]+os[2]))
  i3 = int((loc3 - os[2])/ds[2])
  ax1 = None; ax2 = None; ax3 = None; ax4 = None
  curr_pos = 0
  # Govern the keyboard movement
  j1 = kwargs.get('j1',1); j2 = kwargs.get('j2',1); j3 = kwargs.get('j3',1)

  # Axis labels
  label1 = kwargs.get('label1',' '); label2 = kwargs.get('label2',' '); label3 = kwargs.get('label3', ' ')

  def key_event(e):
    nonlocal i1,loc1,i2,loc2,i3,loc3,ax1,ax2,ax3,ax4,curr_pos

    if e.key=="u" or e.key=="d":
      if e.key=="u":
        i3-=j3
      elif e.key=="d":
        i3+=j3
      i3=i3%ns[2]
      loc3=i3*ds[2]+os[2]

      ax[0,0].cla()

      ax[0,0].imshow(np.flip(data[curr_pos,i3,:,:],0),interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[1],os[1]+(ns[1])*ds[1]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[0,0].set_ylabel(label2,fontsize=kwargs.get('labelsize',14))
      ax[0,0].tick_params(labelsize=kwargs.get('ticksize',14))
      del ax[0,0].lines[:]
      ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
      ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')

      del ax[1,0].lines[:]
      ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
      ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')

      del ax[1,1].lines[:]
      ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
      ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
      ax1.set_yticks([loc3])
      ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
      ax1.tick_params(labelsize=kwargs.get('ticksize',14))
      ax2.set_xticks([loc2])
      ax2.set_xticklabels(['%.2f'%(loc2)])
      ax2.tick_params(labelsize=kwargs.get('ticksize',14))

    elif(e.key=="e" or e.key=="w"):
      if(e.key=="w"):
        i1-=j1
      elif(e.key=="e"):
        i1+=j1
      i1=i1%ns[0]
      loc1=i1*ds[0]+os[0]

      ax[1,1].cla()

      ax[1,1].imshow(data[curr_pos,:,:,i1],interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[1],os[1]+(ns[1])*ds[1],os[2]+(ns[2])*ds[2],os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,1].set_xlabel(label2,fontsize=kwargs.get('labelsize',14))
      ax[1,1].tick_params(labelsize=kwargs.get('ticksize',14))
      del ax[1,1].lines[:]
      ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
      ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
      ax1.set_yticks([loc3])
      ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
      ax1.tick_params(labelsize=kwargs.get('ticksize',14))
      ax2.set_xticks([loc2])
      ax2.set_xticklabels(['%.2f'%(loc2)])
      ax2.tick_params(labelsize=kwargs.get('ticksize',14))

      del ax[1,0].lines[:]
      ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
      ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')

      del ax[0,0].lines[:]
      ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
      ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')
      ax3.set_yticks([loc2])
      ax3.set_yticklabels(['%.2f'%(loc2)],rotation='vertical',va='center')
      ax3.tick_params(labelsize=kwargs.get('ticksize',14))
      ax4.set_xticks([loc1])
      ax4.set_xticklabels(['%.2f'%(loc1)])
      ax4.tick_params(labelsize=kwargs.get('ticksize',14))

    elif e.key=="h" or e.key=="n":
      if e.key=="h":
        i2-=j2
      elif e.key=="n":
        i2+=j2
      i2=i2%ns[1]
      loc2=i2*ds[1]+os[1]

      ax[1,0].cla()

      ax[1,0].imshow(data[curr_pos,:,i2,:],interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[2]+ds[2]*(ns[2]),os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,0].tick_params(labelsize=kwargs.get('ticksize',14))
      del ax[1,1].lines[:]
      ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
      ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')
      ax[1,0].set_xlabel(label1,fontsize=kwargs.get('labelsize',14))
      ax[1,0].set_ylabel(label3,fontsize=kwargs.get('labelsize',14))

      del ax[1,1].lines[:]
      ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
      ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
      ax1.set_yticks([loc3])
      ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
      ax1.tick_params(labelsize=kwargs.get('ticksize',14))
      ax2.set_xticks([loc2])
      ax2.set_xticklabels(['%.2f'%(loc2)])
      ax2.tick_params(labelsize=kwargs.get('ticksize',14))

      del ax[0,0].lines[:]
      ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
      ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')
      ax3.set_yticks([loc2])
      ax3.set_yticklabels(['%.2f'%(loc2)],rotation='vertical',va='center')
      ax3.tick_params(labelsize=kwargs.get('ticksize',14))
      ax4.set_xticks([loc1])
      ax4.set_xticklabels(['%.2f'%(loc1)])
      ax4.tick_params(labelsize=kwargs.get('ticksize',14))

    elif e.key=="left" or e.key=="right" or e.key.isdigit():
      if e.key=="left":
        curr_pos-=1
      if e.key=="right":
        curr_pos+=1
      if e.key.isdigit():
        curr_pos=int(e.key)
      curr_pos=curr_pos%ns[3]

      ax[0,1].cla()
      ax[0,1].get_xaxis().set_visible(False)
      ax[0,1].get_yaxis().set_visible(False)
      ax[0,1].axis('off')
      ax[0,1].text(0.5,0.5,title[curr_pos],horizontalalignment='center',verticalalignment='center',fontsize=50)

      ax[1,0].cla()
      ax[1,0].imshow(data[curr_pos,:,i2,:],interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[2]+ds[2]*(ns[2]),os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
      ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')
      ax[1,0].set_xlabel(label1,fontsize=kwargs.get('labelsize',14))
      ax[1,0].set_ylabel(label3,fontsize=kwargs.get('labelsize',14))

      # yz plane
      ax[1,1].cla()
      ax[1,1].imshow(data[curr_pos,:,:,i1],interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[1],os[1]+(ns[1])*ds[1],os[2]+(ns[2])*ds[2],os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
      ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
      ax[1,1].set_xlabel(label2,fontsize=kwargs.get('labelsize',14))
      ax1.set_yticks([loc3])
      ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
      ax1.tick_params(labelsize=kwargs.get('ticksize',14))
      ax2.set_xticks([loc2])
      ax2.set_xticklabels(['%.2f'%(loc2)])
      ax2.tick_params(labelsize=kwargs.get('ticksize',14))

      # xy plane
      ax[0,0].cla()
      ax[0,0].imshow(np.flip(data[curr_pos,i3,:,:],0),interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[1],os[1]+(ns[1])*ds[1]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
      ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')
      ax[0,0].set_ylabel(label2,fontsize=kwargs.get('labelsize',14))
      ax3.set_yticks([loc2])
      ax3.set_yticklabels(['%.2f'%(loc2)],rotation='vertical',va='center')
      ax3.tick_params(labelsize=kwargs.get('ticksize',14))
      ax4.set_xticks([loc1])
      ax4.set_xticklabels(['%.2f'%(loc1)])
      ax4.tick_params(labelsize=kwargs.get('ticksize',14))

    fig.canvas.draw()

  def onclick(e):
    nonlocal i1,loc1,i2,loc2,i3,loc3,ax1,ax2,ax3,ax4,curr_pos
    tb = plt.get_current_fig_manager().toolbar
    if(tb.mode == ""):
      if e.inaxes==ax1 or e.inaxes==ax2:
        loc2=e.xdata
        i2=int((loc2-os[1])/ds[1])
        loc2=i2*ds[1]+os[1]
        loc3=e.ydata
        i3=int((loc3-os[2])/ds[2])
        loc3=i3*ds[2]+os[2]
      if e.inaxes==ax3 or e.inaxes==ax4:
        loc1=e.xdata
        i1=int((loc1-os[0])/ds[0])
        loc1=i1*ds[0]+os[0]
        loc2=e.ydata
        i2=int((loc2-os[1])/ds[1])
        loc2=i2*ds[1]+os[1]
      if e.inaxes==ax[1,0]:
        loc1=e.xdata
        i1=int((loc1-os[0])/ds[0])
        loc1=i1*ds[0]+os[0]
        loc3=e.ydata
        i3=int((loc3-os[2])/ds[2])
        loc3=i3*ds[2]+os[2]

      ax[1,0].cla()
      ax[1,0].imshow(data[curr_pos,:,i2,:],interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[2]+ds[2]*(ns[2]),os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
      ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')
      ax[1,0].set_xlabel(label1,fontsize=kwargs.get('labelsize',14))
      ax[1,0].set_ylabel(label3,fontsize=kwargs.get('labelsize',14))

      # yz plane
      ax[1,1].cla()
      ax[1,1].imshow(data[curr_pos,:,:,i1],interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[1],os[1]+(ns[1])*ds[1],os[2]+(ns[2])*ds[2],os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
      ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
      ax[1,1].set_xlabel(label2,fontsize=kwargs.get('labelsize',14))
      ax1.set_yticks([loc3])
      ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
      ax1.tick_params(labelsize=kwargs.get('ticksize',14))
      ax2.set_xticks([loc2])
      ax2.set_xticklabels(['%.2f'%(loc2)])
      ax2.tick_params(labelsize=kwargs.get('ticksize',14))

      # xy plane
      ax[0,0].cla()
      ax[0,0].imshow(np.flip(data[curr_pos,i3,:,:],0),interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[1],os[1]+(ns[1])*ds[1]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
      ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')
      ax[0,0].set_ylabel(label2,fontsize=kwargs.get('labelsize',14))
      ax3.set_yticks([loc2])
      ax3.set_yticklabels(['%.2f'%(loc2)],rotation='vertical',va='center')
      ax4.set_xticks([loc1])
      ax4.set_xticklabels(['%.2f'%(loc1)])

      fig.canvas.draw()

  width1 = kwargs.get('width1',4.0); width2 = kwargs.get('width2',4.0); width3 = kwargs.get('width3',4.0)
  widths=[width1,width3]
  heights=[width3,width2]
  gs_kw=dict(width_ratios=widths,height_ratios=heights)
  fig,ax=plt.subplots(2,2,figsize=(width1+width3,width2+width3),gridspec_kw=gs_kw)
  plt.subplots_adjust(wspace=0,hspace=0)
  fig.canvas.mpl_connect('key_press_event', key_event)
  fig.canvas.mpl_connect('button_press_event', onclick)

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

  # Color bar
  if(kwargs.get('cbar',False)):
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.11),
      kwargs.get('wbar',0.02),kwargs.get('hbar',0.78)])
    cbar = fig.colorbar(im,cbar_ax,format='%.2f')
    cbar.ax.tick_params(labelsize=kwargs.get('ticksize',14))
    cbar.set_label(kwargs.get('barlabel',''),fontsize=kwargs.get("barlabelsize",13))
    cbar.draw_all()

  ax[0,1].axis('off')
  plt.show()

