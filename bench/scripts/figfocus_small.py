import numpy as np
import inpout.seppy as seppy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set up io
sep = seppy.sep([])

fsize=18
## Image grid of residual migration
# Read in the image
raxes,rimg = sep.read_file(None,ifname='randrmigtpow.H')
nz  = raxes.n[0]; dz  = raxes.d[0]
nx  = raxes.n[1]; dx  = raxes.d[1]
nro = raxes.n[2]; oro = raxes.o[2]; dro = raxes.d[2]
rimg = rimg.reshape(raxes.n,order='F')
cmin = np.min(rimg); cmax = np.max(rimg)
iro = 0
f,ax = plt.subplots(2,2,figsize=(16,8),gridspec_kw={'width_ratios': [1, 1]})
iros = [3,5,6,7]
strs = ['0.93','0.95','0.96','0.97']
for irow in range(2):
  for icol in range(2):
    ax[irow,icol].imshow(rimg[:,:,iros[iro]],extent=[0,(nx-1)*dx/1000.0,(nz-1)*dz/1000.0,0],cmap='gray',vmin=-0.01,vmax=0.01)
    ax[irow,icol].set_title(r'$\rho = %s$'%strs[iro],fontsize=fsize)
    ax[irow,icol].tick_params(labelsize=fsize)
    if(icol == 0):
      ax[irow,icol].set_ylabel('z (km)',fontsize=fsize)
    else:
      ax[irow,icol].set_yticks([])
    if(irow == 1):
      ax[irow,icol].set_xlabel('x (km)',fontsize=fsize)
    else:
      ax[irow,icol].set_xticks([])
    iro += 1
plt.subplots_adjust(wspace=0.0,hspace=0.5)

plt.savefig('./report/fall2019/fig/rhogridsmall.png',bbox_inches='tight',dpi=150)
plt.show()

