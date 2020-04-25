import inpout.seppy as seppy
import numpy as np
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from resfoc.gain import agc
from resfoc.ssim import ssim
from utils.movie import viewimgframeskey
import matplotlib.pyplot as plt

def mse(img,tgt):
  return np.linalg.norm(img-tgt)

sep = seppy.sep()

# Read in well-focused image
iaxes,img = sep.read_file("fltimg-00760.H")
img = img.reshape(iaxes.n,order='F')
izro = img[:,:,16]

# Read in residual migration image
raxes,res = sep.read_file("zoresfltimg-00760.H")
#raxes,res = sep.read_file("zofftresfltimg-00760.H")
res = res.reshape(raxes.n,order='F')
[nz,nx,nro] = raxes.n; [dz,dx,dro] = raxes.d; [oz,ox,oro] = raxes.o

print(izro.shape,res.shape)

# Perform the patch extraction
nzp = 128; nxp = 128
strdx = 64; strdz = 64
pe = PatchExtractor((nxp,nzp),stride=(strdx,strdz))
ptch = pe.extract(izro.T)
numpx = ptch.shape[0]; numpz = ptch.shape[1]
ptch = ptch.reshape([numpx*numpz,nxp,nzp])

# Plot the patch grid
bgz = 0; egz = (nz)*dz/1000.0; dgz = nzp*dz/1000.0
bgx = 0; egx = (nx)*dx/1000.0; dgx = nxp*dx/1000.0

#fig = plt.figure(figsize=(14,7))
#ax = fig.gca()
#ax.imshow(agc(izro.T).T,extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0],cmap='gray',interpolation='sinc')
#zticks = np.arange(bgz,egz,dgz)
#xticks = np.arange(bgx,egx,dgx)
#ax.set_xticks(xticks)
#ax.set_yticks(zticks)
#ax.grid(linestyle='-',color='k',linewidth=2)

#fig = plt.figure(figsize=(14,7))
#ax = fig.gca()
#ax.imshow(agc(izro.T).T,extent=[0,(nx)*dx/1000.0,(nz)*dz/1000.0,0],cmap='gray',interpolation='sinc')
#zticks = np.arange(bgz+strdz*dz/1000.0,egz-strdz*dz/1000.0+1,dgz)
#xticks = np.arange(bgx+strdx*dx/1000.0,egx-strdx*dx/1000.0+1,dgx)
#ax.set_xticks(xticks)
#ax.set_yticks(zticks)
#ax.grid(which='major',linestyle='-',color='k',linewidth=2)

#viewimgframeskey(ptch,show=False)

# Residual migration patch extraction
nzp = 128; nxp = 128; nrp = nro
strdx = 64; strdz = 64; strdr = nro
per = PatchExtractor((nro,nxp,nzp),stride=(nro,strdx,strdz))
rptch = per.extract(res.T)
print(rptch.shape)
rptch = rptch.reshape([numpx*numpz,nro,nxp,nzp])

# Extract one patch
#viewimgframeskey(rptch[37],show=True)


rhos = np.zeros(nro); sims = np.zeros(nro); mses = np.zeros(nro)

# Loop over each rho and print CWSSISM
pidx = 23
for iro in range(nro):
  rhos[iro] = oro + iro*dro
  sims[iro] = ssim(rptch[pidx,iro],ptch[pidx])
  print("rho=%.8f sim=%.8f"%(rhos[iro],1-sims[iro]))

plt.figure(1)
plt.plot(rhos,1-sims)
plt.xlabel(r'$\rho$')
plt.ylabel('Similarity')

maxsim = np.max(sims)
iro   = np.argmax(sims)

vmin = np.min(ptch[pidx]); vmax = np.max(ptch[pidx])
plt.figure(2)
plt.imshow(ptch[pidx].T,cmap='gray',interpolation='sinc',vmin=vmin,vmax=vmax)
plt.figure(3)
plt.imshow(rptch[pidx,iro].T,cmap='gray',interpolation='sinc',vmin=vmin,vmax=vmax)
plt.title(r'$\rho$=%f'%(rhos[iro]))
plt.figure(4)
plt.imshow(rptch[pidx,9].T,cmap='gray',interpolation='sinc',vmin=vmin,vmax=vmax)
plt.title(r'$\rho$=%f'%(rhos[9]))
plt.show()

