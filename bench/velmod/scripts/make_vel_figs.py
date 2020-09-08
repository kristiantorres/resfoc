import inpout.seppy as seppy
import glob
import numpy as np
import matplotlib.pyplot as plt
from deeplearn.utils import plotseglabel, resample
from utils.ptyprint import progressbar, create_inttag

# Initialize IO
sep = seppy.sep()

# Get all of the models
mdls = sorted(glob.glob("./dat/velflts/velfltmod*.H"))[0:20]
imgs = sorted(glob.glob("./dat/velflts/velfltimg*.H"))[0:20]
lbls = sorted(glob.glob("./dat/velflts/velfltlbl*.H"))[0:20]
refs = sorted(glob.glob("./dat/velflts/velfltref*.H"))[0:20]

fsize = 14
# Loop over all models and create the figures
for ifl in progressbar(range(len(mdls)), "nexamples"):
  # Get the number of the model
  num = int(mdls[ifl].split('velfltmod')[-1].split('.')[0])
  # Read in the velocity model
  vaxes,vel = sep.read_file(ifname=mdls[ifl])
  vel = vel.reshape(vaxes.n,order='F')
  # Read in the image
  iaxes,img = sep.read_file(ifname=imgs[ifl])
  img = img.reshape(iaxes.n,order='F')
  # Read in the label
  laxes,lbl = sep.read_file(ifname=lbls[ifl])
  lbl = lbl.reshape(laxes.n,order='F')
  # Read in the reflectivity
  raxes,ref = sep.read_file(ifname=refs[ifl])
  ref = ref.reshape(raxes.n,order='F')
  if(len(vaxes.n) == 3):
    # Get the axes
    [nz,nx,nex] = vaxes.n
  else:
    # Get the axes
    [nz,nx] = vaxes.n; nex = 1
    vel = np.expand_dims(vel,axis=-1)
    ref = np.expand_dims(ref,axis=-1)
    img = np.expand_dims(img,axis=-1)
    lbl = np.expand_dims(lbl,axis=-1)
  dz = 0.005; dx = 0.010;
  # Get the number of examples in the file
  for iex in range(nex):
    # Plot the velocity model
    fig1 = plt.figure(1,figsize=(8,6)); ax1 = fig1.gca()
    im1 = ax1.imshow(vel[:,:,iex]/1000.0,cmap='jet',interpolation='bilinear',vmin=1.5,vmax=4.5,
              extent=[0.0,(nx)*dx,(nz)*dz,0.0])
    ax1.set_xlabel('X (km)',fontsize=fsize)
    ax1.set_ylabel('Z (km)',fontsize=fsize)
    ax1.tick_params(labelsize=fsize)
    # Colorbar
    cbar_ax = fig1.add_axes([0.92,0.24,0.02,0.51])
    cbar = fig1.colorbar(im1,cbar_ax,format='%.1f')
    cbar.ax.tick_params(labelsize=fsize)
    cbar.set_label('velocity (km/s)',fontsize=fsize)
    cbar.draw_all()
    #plt.show()
    plt.savefig('./fig/velflts/velsnew/vel'+create_inttag(num+iex,10000)+'.png',bbox_inches='tight',dpi=150,transparent=True)
    plt.close()
    # Plot the image
    fig2 = plt.figure(2,figsize=(8,6)); ax2 = fig2.gca()
    im2 = ax2.imshow(img[:,:,iex],cmap='gray',interpolation='sinc',vmin=-3,vmax=3,
        extent=[0.0,(nx)*dx,(nz)*dz,0.0])
    ax2.set_xlabel('X (km)',fontsize=fsize)
    ax2.set_ylabel('Z (km)',fontsize=fsize)
    ax2.tick_params(labelsize=fsize)
    plt.savefig('./fig/velflts/imgsnew/img'+create_inttag(num+iex,10000)+'.png',bbox_inches='tight',dpi=150,transparent=True)
    plt.close()
    # Plot the label on the image
    plotseglabel(img[:,:,iex],lbl[:,:,iex],color='red',
          xlabel='X (km)',ylabel='Z (km)',xmin=0.0,xmax=(nx)*dx,labelsize=fsize,ticksize=fsize,
          zmin=0.0,zmax=(nz)*dz,vmin=-3,vmax=3,interp='sinc',wbox=8,hbox=6,show=False)
    plt.savefig('./fig/velflts/lblsnew/lbl'+create_inttag(num+iex,10000)+'.png',bbox_inches='tight',dpi=150,transparent=True)
    plt.close()
    # Plot the reflectivity
    fig4 = plt.figure(4,figsize=(8,6)); ax4 = fig4.gca()
    im4 = ax4.imshow(ref[:,:,iex],cmap='gray',interpolation='bilinear',vmin=-100,vmax=100,
        extent=[0.0,(nx)*dx,(nz)*dz,0.0])
    ax4.set_xlabel('X (km)',fontsize=fsize)
    ax4.set_ylabel('Z (km)',fontsize=fsize)
    ax4.tick_params(labelsize=fsize)
    plt.savefig('./fig/velflts/refsnew/ref'+create_inttag(num+iex,10000)+'.png',bbox_inches='tight',dpi=150,transparent=True)
    plt.close()

