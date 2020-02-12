"""
Useful functions for creating movies for presentations
@author: Joseph Jennings
@version: 2020.02.06
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
    arr   - input array where the frames are on the fast axis
    odir  - the output directory where to save the frames
    ftype - the type (extension) of figure to save [png]
    qc    - look at each frame as it is saved [False]
    skip  - whether to skip frames [1]
    pttag - write the frame number in pretty (unix-friendly) format
  """
  # Check if directory exists
  if(os.path.isdir(odir) == False):
    os.mkdir(odir)
  nfr = arr.shape[2]
  k = 0
  frames = np.arange(0,nfr,skip)
  for ifr in progressbar(frames, "frames"):
    fig = plt.figure(figsize=(kwargs.get("figsize1",10),kwargs.get("figsize2",10)))
    ax = fig.add_subplot(111)
    ax.imshow(arr[:,:,ifr],cmap=kwargs.get("cmap","gray"),
        extent=[kwargs.get("xmin",0),kwargs.get("xmax",arr.shape[1]),
          kwargs.get("zmax",arr.shape[0]),kwargs.get("zmin",0)],
        vmin=kwargs.get("vmin",np.min(arr[:,:,ifr])),
        vmax=kwargs.get("vmax",np.max(arr[:,:,ifr])))
    ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',18))
    ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',18))
    ax.tick_params(labelsize=kwargs.get('ticksize',18))
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

