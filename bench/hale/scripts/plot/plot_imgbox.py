import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
import subprocess
from genutils.plot import plot_img2d, plot_rhoimg2d
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Read in the image
sep = seppy.sep()
iaxes,img = sep.read_file("spimgbobang.H")
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)[:,0,:,:]
#stkw = agc(np.sum(img,axis=1))[30:542,:512]
stkw = np.sum(img,axis=1)[30:542,:512]
dz,da,dy,dx = iaxes.d; oz,oa,oy,ox = iaxes.o
nx,nz = stkw.shape

#sep.write_file("presmooth.H",stkw.T)
#sp = subprocess.check_call("python scripts/SOSmoothing.py -fin presmooth.H -fout smooth.H",shell=True)
saxes,smt = sep.read_file("smooth.H")
smt = np.ascontiguousarray(smt.reshape(saxes.n,order='F'))
baxes,smb = sep.read_file("smooth-smb.H")
smb = np.ascontiguousarray(smb.reshape(baxes.n,order='F')).T

#plot_img2d(stkw.T,dx=dx,dz=dz,ox=30*dx+ox,pclip=0.5,aspect=3.0,figname='./fig/beiimg.png')
#plot_img2d(smt,dx=dx,dz=dz,ox=30*dx+ox,pclip=0.5,aspect=3.0,figname='./fig/beiimgsmth.png')

vmin,vmax = np.min(stkw), np.max(stkw)
fig = plt.figure(figsize=(10,5)); ax = fig.gca()
ax.imshow(stkw.T,extent=[30*dx+ox,ox+30*dx+nx*dx,nz*dz,0],interpolation='bilinear',
          vmin=vmin*0.5,vmax=vmax*0.5,cmap='gray',aspect=3.0)
im = ax.imshow(smb.T,extent=[30*dx+ox,ox+30*dx+nx*dx,nz*dz,0],interpolation='bilinear',
          vmin=0.0,vmax=1.0,aspect=3.0,alpha=0.1,cmap='jet')
cbar_ax = fig.add_axes([0.74,0.15,0.02,0.70])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.solids.set(alpha=1)
cbar.ax.tick_params(labelsize=15)
cbar.set_label('Semblance',fontsize=15)
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Z (km)',fontsize=15)
ax.tick_params(labelsize=15)
plt.savefig("./fig/beisemb.png",dpi=150,bbox_inches='tight',transparent=True)
#plt.show()

#rect = patches.Rectangle((ox+30*dx,100*dz),nx*dx,256*dz,linewidth=2,edgecolor='yellow',facecolor='none')
#ax.add_patch(rect)
#plt.savefig("./fig/beibox.png",dpi=150,bbox_inches='tight',transparent=True)
#plt.show()

