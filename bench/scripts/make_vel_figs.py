import inpout.seppy as seppy
import glob
import numpy as np
import matplotlib.pyplot as plt
from deeplearn.utils import plotseglabel, resample
from utils.ptyprint import progressbar, create_inttag

# Initialize IO
sep = seppy.sep([])

# Get all of the models
mdls = sorted(glob.glob("./dat/velflts/velfltmod*.H"))[0:2]
imgs = sorted(glob.glob("./dat/velflts/velfltimg*.H"))[0:2]
lbls = sorted(glob.glob("./dat/velflts/velfltlbl*.H"))[0:2]

fsize = 18
# Loop over all models and create the figures
for ifl in progressbar(range(len(mdls)), "nexamples"):
  # Get the number of the model
  num = int(mdls[ifl].split('velfltmod')[-1].split('.')[0])
  # Read in the velocity model
  vaxes,vel = sep.read_file(None,ifname=mdls[ifl])
  vel = vel.reshape(vaxes.n,order='F')
  # Read in the image
  iaxes,img = sep.read_file(None,ifname=imgs[ifl])
  img = img.reshape(iaxes.n,order='F')
  # Read in the label
  laxes,lbl = sep.read_file(None,ifname=lbls[ifl])
  lbl = lbl.reshape(laxes.n,order='F')
  # Get the axes
  nz = vaxes.n[0]; dz = 5/1000.0#vaxes.d[0]/1000.0
  nx = vaxes.n[1]; dx = 10/1000.0#vaxes.d[1]/1000.0
  # Get the number of examples in the file
  if(len(vaxes.n) == 3):
    nex = vaxes.n[2]
  else:
    nex = 1
    vel = np.expand_dims(vel,axis=-1)
    img = np.expand_dims(img,axis=-1)
    lbl = np.expand_dims(lbl,axis=-1)
  for iex in range(nex):
    # Plot the velocity model
    fig1 = plt.figure(1,figsize=(14,7)); ax1 = fig1.gca()
    im1 = ax1.imshow(vel[:,:,iex]/1000.0,cmap='jet',interpolation='bilinear',vmin=1.5,vmax=4.5,
              extent=[0.0,(nx-1)*dx,(nz-1)*dz,0.0])
    ax1.set_xlabel('X (km)',fontsize=fsize)
    ax1.set_ylabel('Z (km)',fontsize=fsize)
    ax1.tick_params(labelsize=fsize)
    # Colorbar
    cbar_ax = fig1.add_axes([0.91,0.11,0.02,0.77])
    cbar = fig1.colorbar(im1,cbar_ax,format='%.2f')
    cbar.ax.tick_params(labelsize=fsize)
    cbar.set_label('velocity (km/s)',fontsize=fsize)
    cbar.draw_all()
    plt.savefig('./fig/velflts/vels/vel'+create_inttag(num+iex,10000)+'.png',bbox_inches='tight',dpi=150)
    plt.close()
    # Plot the image
    fig2 = plt.figure(2,figsize=(14,7)); ax2 = fig2.gca()
    im2 = ax2.imshow(img[:,:,iex],cmap='gray',interpolation='sinc',vmin=-3,vmax=3,
        extent=[0.0,(nx-1)*dx,(nz-1)*dz,0.0])
    ax2.set_xlabel('X (km)',fontsize=fsize)
    ax2.set_ylabel('Z (km)',fontsize=fsize)
    ax2.tick_params(labelsize=fsize)
    plt.savefig('./fig/velflts/imgs/img'+create_inttag(num+iex,10000)+'.png',bbox_inches='tight',dpi=150)
    plt.close()
    # Plot the label on the image
    plotseglabel(img[:,:,iex],lbl[:,:,iex],color='red',
          xlabel='X (km)',ylabel='Z (km)',xmin=0.0,xmax=(nx-1)*dx,
          zmin=0.0,zmax=(nz-1)*dz,vmin=-3,vmax=3,interp='sinc',fsize1=14,fsize2=7,show=False)
    plt.savefig('./fig/velflts/lbls/lbl'+create_inttag(num+iex,10000)+'.png',bbox_inches='tight',dpi=150)
    plt.close()

