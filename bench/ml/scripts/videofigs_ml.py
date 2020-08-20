import os
import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from scaas.trismooth import smooth
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from deeplearn.keraspredict import segmentfaults
from deeplearn.utils import plotsegprobs
from deeplearn.dataloader import load_all_unlabeled_data,load_labeled_flat_data
import matplotlib.pyplot as plt
from genutils.plot import plot_cubeiso
from numba import cuda

sep = seppy.sep()

# Well-focused image
faxes,fog = sep.read_file("../focdat/dat/focdefoc/mltestfog.H")
fog = np.ascontiguousarray(fog.reshape(faxes.n,order='F').T).astype('float32')

# Unfocused image
daxes,dog = sep.read_file("../focdat/dat/focdefoc/mltestdog.H")
dog = np.ascontiguousarray(dog.reshape(daxes.n,order='F').T).astype('float32')

# Training focused image
tfaxes,tfog = sep.read_file("./dat/fog-0033.H")
tfog = np.ascontiguousarray(tfog.reshape(tfaxes.n,order='F').T).astype('float32')
zotfog = agc(tfog[16])

# Training defocused image
tdaxes,tdog = sep.read_file("./dat/dog-0033.H")
tdog = np.ascontiguousarray(tdog.reshape(tdaxes.n,order='F').T).astype('float32')
zotdog = agc(tdog[16])

# Training residually focused image
traxes,trog = sep.read_file("./dat/reso-0033.H")
trog = np.ascontiguousarray(trog.reshape(traxes.n,order='F').T).astype('float32')
zotrog = agc(trog[16])

[nz,nx,nh] = faxes.n; [dz,dx,dh] = faxes.d;

dz /= 1000.0; dx /= 1000.0

zofog = fog[16]
fogg = agc(zofog).T

zodog = dog[16]
dogg = agc(zodog).T

nx = 400; fx = 305
nz = 300; fz = 120

## Set the GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
#tf.compat.v1.GPUOptions(allow_growth=True)
#
## Read in the fault focusing network
#with open("./dat/networks/xwarch-gpu10.json",'r') as f:
#  fltmdl = model_from_json(f.read())
#fltmdl.load_weights("/net/fantastic/scr2/joseph29/xwarch-gpu10ep-chkpnt.h5")
#
#fltprbf = segmentfaults(fogg,fltmdl,nzp=128,nxp=128,strdz=64,strdx=64)
#fltprbd = segmentfaults(dogg,fltmdl,nzp=128,nxp=128,strdz=64,strdx=64)
#
#foggw    = fogg[fz:fz+nz,fx:fx+nx]
#fltprbfw = fltprbf[fz:fz+nz,fx:fx+nx]
#
#doggw    = dogg[fz:fz+nz,fx:fx+nx]
#fltprbdw = fltprbd[fz:fz+nz,fx:fx+nx]
#
#plotsegprobs(foggw,fltprbfw,pmin=0.5,interp='sinc',vmin=-2.5,vmax=2.5,
#             wbox=10,hbox=10,barx=0.925,barz=0.205,hbar=0.58,xlabel='X (km)',ylabel='Z (km)',
#             xmin=fx*dx,xmax=(fx+nx)*dx,zmin=fz*dz,zmax=(fz+nz)*dz,
#             labelsize=16,ticksize=16,fname='./fig/videofigs/fogfltprb',ftype='png',cropsize=170)
#
#plotsegprobs(doggw,fltprbdw,pmin=0.5,interp='sinc',vmin=-2.5,vmax=2.5,
#             wbox=10,hbox=10,barx=0.925,barz=0.205,hbar=0.58,xlabel='X (km)',ylabel='Z (km)',
#             xmin=fx*dx,xmax=(fx+nx)*dx,zmin=fz*dz,zmax=(fz+nz)*dz,
#             labelsize=16,ticksize=16,fname='./fig/videofigs/dogfltprb',ftype='png',cropsize=170)
#
#cuda.close()

# Window the training images
#nx = 512; fx = 256
#nz = 256; fz = 138
##
#nzp = 64; nxp = 64
#bgz = 0.0; egz = (nz+1)*dz; dgz = nzp*dz
#bgx = 0.0; egx = (nx+1)*dx; dgx = nxp*dx
#zticks = np.arange(bgz,egz,dgz)
#xticks = np.arange(bgx,egx,dgx)
#
fsize = 15
#fig = plt.figure(figsize=(10,10)); ax = fig.gca()
#ax.imshow(zotfog[fx:fx+nx,fz:fz+nz].T,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5,
#          extent=[0.0,(nx)*dx,(nz)*dz,0.0])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.set_xticks(xticks)
#ax.set_yticks(zticks)
#ax.grid(linestyle='-',color='k',linewidth=2)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/videofigs/foctrainimg.png',dpi=150,transparent=True,bbox_inches='tight')
#plt.close()
#
#fig = plt.figure(figsize=(10,10)); ax = fig.gca()
#ax.imshow(zotdog[fx:fx+nx,fz:fz+nz].T,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5,
#          extent=[0.0,(nx)*dx,(nz)*dz,0.0])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.set_xticks(xticks)
#ax.set_yticks(zticks)
#ax.grid(linestyle='-',color='k',linewidth=2)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/videofigs/deftrainimg.png',dpi=150,transparent=True,bbox_inches='tight')
#plt.close()
#
#fig = plt.figure(figsize=(10,10)); ax = fig.gca()
#ax.imshow(zotrog[fx:fx+nx,fz:fz+nz].T,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5,
#          extent=[0.0,(nx)*dx,(nz)*dz,0.0])
#ax.set_xlabel('X (km)',fontsize=fsize)
#ax.set_ylabel('Z (km)',fontsize=fsize)
#ax.set_xticks(xticks)
#ax.set_yticks(zticks)
#ax.grid(linestyle='-',color='k',linewidth=2)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/videofigs/restrainimg.png',dpi=150,transparent=True,bbox_inches='tight')
#plt.close()

# Load all data
#numex=5040
#nimgs = int(numex/105)
#focdat = load_all_unlabeled_data("/scr1/joseph29/angfaultfocptch.h5",0,nimgs)
#resdat = load_all_unlabeled_data("/scr1/joseph29/angfaultresptch.h5",0,nimgs)
#defdat,deflbl = load_labeled_flat_data("/scr1/joseph29/alldefocangs.h5",None,0,numex)

#print(focdat.shape,resdat.shape,defdat.shape)

## QC the images
#os = [-70.0,0.0,0.0]; ds = [2.22,0.01,0.01]
#for iex in range(100):
#  idx = np.random.randint(420)
#  plot_cubeiso(focdat[idx,:,:,:,0],os=os,ds=ds,elev=15,verb=False,show=False,
#      x1label='\nX (km)',x2label='\nAngle '+r'($\degree$)',x3label='Z (km)',stack=True,
#      figname="./fig/videofigs/foc%d.png"%(iex))
#  plot_cubeiso(defdat[idx,:,:,:,0],os=os,ds=ds,elev=15,verb=False,show=False,
#      x1label='\nX (km)',x2label='\nAngle '+r'($\degree$)',x3label='Z (km)',stack=True,
#      figname="./fig/videofigs/def%d.png"%(iex))
#  plot_cubeiso(resdat[idx,:,:,:,0],os=os,ds=ds,elev=15,verb=False,show=False,
#      x1label='\nX (km)',x2label='\nAngle '+r'($\degree$)',x3label='Z (km)',stack=True,
#      figname="./fig/videofigs/res%d.png"%(iex))


# Plot the training and accuracy
#_,loss = sep.read_file("./dat/nnhistory/angfltfocdefoc_loss.H")
#_,vlss = sep.read_file("./dat/nnhistory/angfltfocdefoc_vlss.H")
#_,accu = sep.read_file("./dat/nnhistory/angfltgfocdefoc_accu.H")
#_,vacu = sep.read_file("./dat/nnhistory/angfltfocdefoc_vacu.H")
#
#fsize = 18
#fig = plt.figure(1,figsize=(10,10)); ax = fig.gca()
#lin1 = plt.plot(loss/np.max(loss),label='Training',linewidth=2)
#lin2 = plt.plot(vlss/np.max(loss),label='Validation',linewidth=2)
#ax.legend(fontsize=fsize)
#ax.set_xlabel('Epochs',fontsize=fsize)
#ax.set_ylabel('Normalized loss',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/videofigs/loss.png',dpi=150,transparent=True,bbox_inches='tight')
#plt.close()
#
#fig = plt.figure(1,figsize=(10,10)); ax = fig.gca()
#lin1 = plt.plot(accu,label='Training',linewidth=2)
#lin2 = plt.plot(vacu,label='Validation',linewidth=2)
#ax.legend(fontsize=fsize)
#ax.set_xlabel('Epochs',fontsize=fsize)
#ax.set_ylabel('Accuracy (%)',fontsize=fsize)
#ax.tick_params(labelsize=fsize)
#plt.savefig('./fig/videofigs/accu.png',dpi=150,transparent=True,bbox_inches='tight')
#plt.close()

raxes,rimg = sep.read_file("rhoangcnn3.H")
rimg = rimg.reshape(raxes.n,order='F')

paxes,prb = sep.read_file("focprb.H")
#prb = smooth(np.log(np.abs(prb.reshape(paxes.n,order='F').T + 0.00001)).astype('float32'),rect1=40,rect2=40)
prb = prb.reshape(paxes.n,order='F').astype('float32').T
prb = smooth(prb,rect1=40,rect2=40)

[nz,nx,nro] = paxes.n; [dz,dx,dro] = paxes.d; [oz,ox,oro] = paxes.o

saxes,stk = sep.read_file("../focdat/dat/refocus/mltest/mltestdogstk2.H")
stk = np.ascontiguousarray(stk.reshape(saxes.n,order='F').T).astype('float32')

stkg = agc(stk)

fsize=16
fxl = 49; nx = 400
fzl = 120; nz = 300
for iro in range(nro):
  rho = oro + iro*dro
  fig = plt.figure(figsize=(10,10)); ax = fig.gca()
  ax.imshow(stkg[iro,fxl:fxl+nx,fzl:fzl+nz].T,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5,
                 extent=[fx*dx/1000.0,(fx+nx)*dx/1000.0,(fz+nz)*dz/1000.0,fz*dz/1000.0])
  im = ax.imshow(prb[iro,fxl:fxl+nx,fzl:fzl+nz].T,cmap='jet',interpolation='bilinear',alpha=0.2,vmin=np.min(prb),vmax=np.max(prb),
                 extent=[fx*dx/1000.0,(fx+nx)*dx/1000.0,(fz+nz)*dz/1000.0,fz*dz/1000.0])
#  ax.set_title(r'$\rho$=%.5f'%(rho),fontsize=fsize)
  ax.set_xlabel('X (km)',fontsize=fsize)
  ax.set_ylabel('Z (km)',fontsize=fsize)
  ax.tick_params(labelsize=fsize)
  cbar_ax2 = fig.add_axes([0.925,0.205,0.02,0.58])
  cbar2 = fig.colorbar(im,cbar_ax2,format='%.2f')
  cbar2.solids.set(alpha=0.5)
  cbar2.ax.tick_params(labelsize=fsize)
  cbar2.set_label(r'Focus probability',fontsize=fsize)
  plt.savefig('./fig/videofigs/focprbs/focprb%d.png'%(iro),dpi=150,transparent=True,bbox_inches='tight')
  plt.close()
#
#  #plt.figure(); plt.imshow(rimg.T,cmap='seismic')
#  #plt.show()

#oaxes,ro1 = sep.read_file("focprbrho1full.H")
#ro1 = ro1.reshape(oaxes.n,order='F')
#
#plt.imshow(np.log(np.abs(ro1/np.max(ro1))),cmap='jet')
#plt.show()

