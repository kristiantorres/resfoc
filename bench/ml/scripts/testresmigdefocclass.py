import os
import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from deeplearn.keraspredict import focdefocang
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.utils import normalize
from scaas.trismooth import smooth
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in the image
raxes,res = sep.read_file("../focdat/dat/refocus/mltest/mltestfogangwind.H")
res = res.reshape(raxes.n,order='F')
res = np.ascontiguousarray(res.T).astype('float32') # [nro,nx,na,nz]
[nz,na,nx,nro] = raxes.n; [dz,da,dx,dro] = raxes.d; [oz,da,ox,oro] = raxes.o
stk = np.sum(res,axis=2)

## Apply AGC
# Angle gathers
gres =  np.asarray(Parallel(n_jobs=24)(delayed(agc)(res[iro]) for iro in range(nro)))
gimgt = np.ascontiguousarray(np.transpose(gres,(0,2,3,1))) # [nro,nx,na,nz] -> [nro,na,nz,nx]
# Stack
stkg = agc(stk)
stkgt = np.transpose(stkg,(0,2,1)) # [nro,nx,nz] -> [nro,nz,nx]

os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

# Read in the fault focusing network
with open('./dat/networks/angfltfocdefoc-arch.json','r') as f:
  focmdl = model_from_json(f.read())
focmdl.load_weights('/scr1/joseph29/angfltfocdefoc-chkpnt.h5')

focmdl.summary()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Output
pred = np.zeros([nro,nz,nx],dtype='float32')

# Build the Patch Extractors
na = 64; nzp = 64; nxp = 64; strdz = 32; strdx = 32
pea = PatchExtractor((na,nzp,nxp),stride=(na,strdz,strdx))
dummy = np.squeeze(pea.extract(gimgt[0]))
numpz = dummy.shape[0]; numpx = dummy.shape[1]

allptchs = np.zeros([nro,numpz,numpx,nzp,nxp])

fxl =   0; nxl = nx
fzl = 120; nzl = 300
fsize = 16
# Predict for each rho
for iro in range(nro):
  aptch = np.squeeze(pea.extract(gimgt[iro]))
  # Flatten patches and make a prediction on each
  numpz = aptch.shape[0]; numpx = aptch.shape[1]
  aptchf = np.expand_dims(normalize(aptch.reshape([numpz*numpx,na,nzp,nxp])),axis=-1)
  focprd = focmdl.predict(aptchf)

  focprdptch = np.zeros([numpz*numpx,nzp,nxp])
  for iptch in range(numpz*numpx): focprdptch[iptch,:,:] = focprd[iptch]
  focprdptch = focprdptch.reshape([numpz,numpx,nzp,nxp])
  allptchs[iro] = focprdptch

  # Output probabilities
  per = PatchExtractor((nzp,nxp),stride=(strdz,strdx))
  focprdimg = np.zeros([nz,nx])
  _ = per.extract(focprdimg)

  pred[iro] = per.reconstruct(focprdptch)


predt = np.transpose(pred,(0,2,1))

for iro in range(nro):
  # Make plot
  rho = oro + iro*dro
  fig = plt.figure(figsize=(10,10)); ax = fig.gca()
  ax.imshow(stkg[iro,fxl:fxl+nx,fzl:fzl+nz].T,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5,
                 extent=[fxl*dx/1000.0,(fxl+nxl)*dx/1000.0,(fzl+nzl)*dz/1000.0,fzl*dz/1000.0])
  im = ax.imshow(predt[iro,fxl:fxl+nx,fzl:fzl+nz].T,cmap='jet',interpolation='bilinear',alpha=0.2,vmin=np.min(pred),vmax=np.max(pred),
                 extent=[fxl*dx/1000.0,(fxl+nxl)*dx/1000.0,(fzl+nzl)*dz/1000.0,fzl*dz/1000.0])
  ax.set_title(r'$\rho$=%.5f'%(rho),fontsize=fsize)
  ax.set_xlabel('X (km)',fontsize=fsize)
  ax.set_ylabel('Z (km)',fontsize=fsize)
  ax.tick_params(labelsize=fsize)
  cbar_ax2 = fig.add_axes([0.925,0.205,0.02,0.58])
  cbar2 = fig.colorbar(im,cbar_ax2,format='%.2f')
  cbar2.solids.set(alpha=0.5)
  cbar2.ax.tick_params(labelsize=fsize)
  cbar2.set_label(r'Focus probability',fontsize=fsize)
  plt.savefig('./fig/resfocclass/focprb%d.png'%(iro),dpi=150,transparent=True,bbox_inches='tight')
  plt.close()


