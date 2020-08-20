import os
import inpout.seppy as seppy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from numba import cuda
from resfoc.estro import anglemask, estro_fltangfocdefoc, refocusimg
from resfoc.gain import agc
from joblib import Parallel, delayed
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey

sep = seppy.sep()

iaxes,img= sep.read_file("../focdat/dat/refocus/mltest/mltestdogang2.H")
img = img.reshape(iaxes.n,order='F')
img = np.ascontiguousarray(img.T).astype('float32') # [nro,nx,na,nz]
stk = np.sum(img,axis=2)
## Apply AGC
# Angle gathers
[nz,na,nx,nro] = iaxes.n; [dz,da,dx,dro] = iaxes.d; [oz,da,ox,oro] = iaxes.o
gimg =  np.asarray(Parallel(n_jobs=24)(delayed(agc)(img[iro]) for iro in range(nro)))
gimgt = np.ascontiguousarray(np.transpose(gimg,(0,2,3,1))) # [nro,nx,na,nz] -> [nro,na,nz,nx]
# Stack
stkg = agc(stk)
stkgt = np.transpose(stkg,(0,2,1)) # [nro,nx,nz] -> [nro,nz,nx]

# Create and apply angle mask
mask = anglemask(nz,na,zpos=0.05,apos=0.6,mode='slant',rand=True,rectz=10,recta=10).T

#plt.figure(1)
#plt.imshow(mask.T,cmap='gray')
#plt.show()

# Replicate along the spatial dimension
maskrep = np.repeat(mask[np.newaxis,:,:],nx,axis=0)

# Apply the mask to each rho
masked  = np.asarray([maskrep*gimg[iro] for iro in range(nro)])
maskedt = np.ascontiguousarray(np.transpose(masked,(0,2,3,1))) # [nro,nx,na,nz] -> [nro,na,nz,nx]

#plt.figure(2)
#plt.imshow(gimg[20,300,:,:].T,cmap='gray',interpolation='sinc')
#plt.figure(3)
#plt.imshow(masked[20,300,:,:].T,cmap='gray',interpolation='sinc')
#plt.show()

# Set GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
tf.compat.v1.GPUOptions(allow_growth=True)

# Read in the fault focusing network
with open('./dat/networks/angfltfocdefoc-arch.json','r') as f:
  focmdl = model_from_json(f.read())
focmdl.load_weights('/scr1/joseph29/angfltfocdefoc-chkpnt.h5')

focmdl.summary()

rho,angfocs = estro_fltangfocdefoc(gimgt,focmdl,dro,oro,rectz=40,rectx=40)
rhomask,angfocsmask = estro_fltangfocdefoc(maskedt,focmdl,dro,oro,rectz=40,rectx=40)
cuda.close()

rfi    = refocusimg(stkgt,rho,dro)
rfimsk = refocusimg(stkgt,rhomask,dro)

# Defocused image
rho1img = stkgt[21]

fsize=15
fig3 = plt.figure(2,figsize=(8,8)); ax3 = fig3.gca()
ax3.imshow(rho1img,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
im3 = ax3.imshow(rho,cmap='seismic',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear',vmin=0.98,vmax=1.02,alpha=0.1)
ax3.set_xlabel('X (km)',fontsize=fsize)
ax3.set_ylabel('Z (km)',fontsize=fsize)
ax3.tick_params(labelsize=fsize)
ax3.set_title(r"Full",fontsize=fsize)
cbar_ax3 = fig3.add_axes([0.91,0.15,0.02,0.70])
cbar3 = fig3.colorbar(im3,cbar_ax3,format='%.2f')
cbar3.solids.set(alpha=1)
cbar3.ax.tick_params(labelsize=fsize)
cbar3.set_label(r'$\rho$',fontsize=fsize)
#plt.savefig('./fig/rhoangimg.png',transparent=True,dpi=150,bbox_inches='tight')
#plt.close()

fig4 = plt.figure(3,figsize=(8,8)); ax4 = fig4.gca()
ax4.imshow(rho1img,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
im4 = ax4.imshow(rhomask,cmap='seismic',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear',vmin=0.98,vmax=1.02,alpha=0.1)
ax4.set_xlabel('X (km)',fontsize=fsize)
ax4.set_ylabel('Z (km)',fontsize=fsize)
ax4.tick_params(labelsize=fsize)
ax4.set_title(r"Mask",fontsize=fsize)
cbar_ax4 = fig4.add_axes([0.91,0.15,0.02,0.70])
cbar4 = fig4.colorbar(im4,cbar_ax4,format='%.2f')
cbar4.solids.set(alpha=1)
cbar4.ax.tick_params(labelsize=fsize)
cbar4.set_label(r'$\rho$',fontsize=fsize)
#plt.savefig('./fig/rhoangimg.png',transparent=True,dpi=150,bbox_inches='tight')
#plt.close()

# Plot the refocused images
fig5 = plt.figure(4,figsize=(8,8)); ax5 = fig5.gca()
ax5.imshow(rfi,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax5.set_xlabel('X (km)',fontsize=fsize)
ax5.set_ylabel('Z (km)',fontsize=fsize)
ax5.set_title('Full',fontsize=fsize)
ax5.tick_params(labelsize=fsize)
#plt.savefig('./fig/rfisemb.png',transparent=True,dpi=150,bbox_inches='tight')
#plt.close()

fig6 = plt.figure(5,figsize=(8,8)); ax6 = fig6.gca()
ax6.imshow(rfimsk,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=-2.5,vmax=2.5)
ax6.set_xlabel('X (km)',fontsize=fsize)
ax6.set_ylabel('Z (km)',fontsize=fsize)
ax6.set_title('Mask',fontsize=fsize)
ax6.tick_params(labelsize=fsize)
#plt.savefig('./fig/rfiang.png',transparent=True,dpi=150,bbox_inches='tight')
#plt.close()

plt.show()
