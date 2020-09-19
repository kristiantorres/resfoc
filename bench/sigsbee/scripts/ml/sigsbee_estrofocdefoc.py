import os
import inpout.seppy as seppy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from resfoc.gain import agc
from resfoc.estro import estro_fltangfocdefoc, refocusimg
from scaas.velocity import salt_mask
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey

sep = seppy.sep()
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

# Read in the residually migrated image
#iaxes,img = sep.read_file("resmskoverwt.H")
iaxes,img = sep.read_file("sigsbeewrngposrest.H")
[nz,na,nx,nro] = iaxes.n; [dz,da,dx,dro] = iaxes.d; [oz,oa,ox,oro] = iaxes.o
img  = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')
imgt = np.ascontiguousarray(np.transpose(img,(0,2,3,1)))

nzp = 64; nxp = 64
strdz = int(nzp/2 + 0.5)
strdx = int(nxp/2 + 0.5)

# Define window
bxw = 20;  exw = 480
bzw = 240; ezw = 1150

# Window and stack
imgw = imgt[:,:,bzw:ezw,bxw:exw]
stk  = np.sum(imgw[:,32:],axis=1)
#imgw = imgt[:,32:,bzw:ezw,bxw:exw]
stk  = np.sum(imgw,axis=1)

per = PatchExtractor((nro,nzp,nxp),stride=(nro,nzp//2,nxp//2))
stkw = per.reconstruct(per.extract(stk))
sc2 = 0.2
kmin = sc2*np.min(stkw); kmax= sc2*np.max(stkw)
print(stkw.shape)

viewimgframeskey(stkw,transp=False,pclip=0.4)

# Read in the velocity model to mask the salt
vaxes,vel = sep.read_file("sigoverw_velint.H")
vel = vel.reshape(vaxes.n,order='F').T
velw = vel[bxw:exw,0,bzw:ezw].T
pev = PatchExtractor((nzp,nxp),stride=(nzp//2,nxp//2))
velww = pev.reconstruct(pev.extract(velw))

# Read in the fault focusing network
with open('./dat/focangarc.json','r') as f:
  focmdl = model_from_json(f.read())
focwgt = focmdl.load_weights('/scr1/joseph29/sigsbee_focangnoreschkpt.h5')
#focwgt = focmdl.load_weights('./dat/focangwgts.h5')

verb = False
if(verb): focmdl.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

rho,fltfocs = estro_fltangfocdefoc(imgw,focmdl,dro,oro,rectz=30,rectx=30,verb=True)

print(rho.shape,velww.shape)

# Mask the rho image
msk,rhomsk = salt_mask(rho,velww,saltvel=4.5)
idx = rhomsk == 0.0
rhomsk[idx] = 1.0

# Plot rho on top of stack
fsize = 15
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(stkw[20],interpolation='bilinear',cmap='gray',vmin=kmin,vmax=kmax,aspect='auto')
im = ax.imshow(rhomsk,cmap='seismic',interpolation='bilinear',vmin=0.975,vmax=1.025,alpha=0.2,aspect='auto')
cbar_ax = fig.add_axes([0.925,0.212,0.02,0.58])
cbar = fig.colorbar(im,cbar_ax,format='%.2f')
cbar.solids.set(alpha=1)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r'$\rho$',fontsize=fsize)
plt.show()

print(stkw.shape,rhomsk.shape)
# Refocus the stack
rfi = refocusimg(stkw,rhomsk,dro)

sep.write_file("sigfocrhopos.H",rhomsk,ds=[dz,dx])
sep.write_file("sigfocrfipos.H",rfi,ds=[dz,dx])
sep.write_file("stkfocwindpos.H",stkw[20],ds=[dz,dx])

